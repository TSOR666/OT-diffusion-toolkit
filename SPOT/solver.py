"""Core ProductionSPOTSolver implementation."""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from ._version import __version__
from .config import SolverConfig
from .constants import (
    DEFAULT_GPU_MEMORY_FRACTION,
    EPSILON_CLAMP,
    EPSILON_MIN,
    MAX_BLOCK_SIZE,
    MIN_BLOCK_SIZE,
    ROW_SUM_MIN,
)
from .dpm_solver import DPMSolverPP
from .logger import logger
from .results import SamplingResult, StepContext
from .schedules import CosineSchedule, LinearSchedule, NoiseScheduleProtocol
from .sinkhorn import OptimizedSinkhornKernel
from .transport import blockwise_soft_assignment, make_grid_patch_transport

# Import new integrators and correctors (optional, gracefully degrade if not available)
try:
    from .integrators import (
        HeunIntegrator,
        DDIMIntegrator,
        AdaptiveIntegrator,
        ExponentialIntegrator,
    )
    INTEGRATORS_AVAILABLE = True
except ImportError:
    INTEGRATORS_AVAILABLE = False
    HeunIntegrator = None
    DDIMIntegrator = None
    AdaptiveIntegrator = None
    ExponentialIntegrator = None

try:
    from .corrector import (
        LangevinCorrector,
        TweedieCorrector,
        AdaptiveCorrector,
    )
    CORRECTORS_AVAILABLE = True
except ImportError:
    CORRECTORS_AVAILABLE = False
    LangevinCorrector = None
    TweedieCorrector = None
    AdaptiveCorrector = None

__all__ = ["ProductionSPOTSolver"]


class ProductionSPOTSolver:
    """Main solver with all critical fixes applied.
    
    Note: The score model is automatically set to eval mode and its parameters
    are frozen (requires_grad=False) for inference. If you need to reuse the
    model for training, re-enable gradients with model.train() and 
    model.requires_grad_(True).
    """
    
    def __init__(
        self,
        score_model,
        noise_schedule: Optional[NoiseScheduleProtocol] = None,
        config: Optional[SolverConfig] = None,
        device: Optional[torch.device] = None,
        compute_dtype: Optional[torch.dtype] = None
    ):
        """Initialize SPOT solver with unified schedule semantics.
        
        Args:
            score_model: Neural network score model (must accept (x, t) and return same shape as x).
                        Model will be frozen for inference (requires_grad=False).
            noise_schedule: Noise schedule protocol (default: CosineSchedule)
            config: Solver configuration
            device: Compute device (default: auto-detect)
            compute_dtype: Precision for computations (default: fp32 for stability)
        """
        self.config = config or SolverConfig()
        self.config.validate()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = compute_dtype or torch.float32

        self._memory_format = (
            torch.channels_last
            if self.config.use_channels_last and self.device.type == 'cuda'
            else torch.contiguous_format
        )

        autocast_enabled = self.config.use_mixed_precision and self.device.type == 'cuda'
        autocast_dtype = None
        if autocast_enabled:
            if self.config.autocast_precision == 'none' or self.config.autocast_precision == 'fp32':
                autocast_enabled = False
            elif self.config.autocast_precision == 'bf16':
                if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    autocast_dtype = torch.bfloat16
                else:
                    logger.debug("bf16 autocast requested but not supported; falling back to fp16")
                    autocast_dtype = torch.float16
            elif self.config.autocast_precision == 'fp16':
                autocast_dtype = torch.float16

        self._autocast_kwargs: Dict[str, Any] = {
            'device_type': 'cuda' if self.device.type == 'cuda' else 'cpu',
            'enabled': bool(autocast_enabled)
        }
        if autocast_dtype is not None and self._autocast_kwargs['device_type'] == 'cuda' and self._autocast_kwargs['enabled']:
            self._autocast_kwargs['dtype'] = autocast_dtype
        self._autocast_dtype = autocast_dtype

        if not isinstance(score_model, torch.nn.Module):
            raise TypeError(f"score_model must be a torch.nn.Module, got {type(score_model)}")

        self._original_tf32_matmul = None
        self._original_tf32_cudnn = None
        self._original_matmul_prec = None
        self._tf32_enabled = False

        if self.config.enable_tf32 and torch.cuda.is_available():
            logger.debug("TF32 opted in; will apply within tf32_context during sampling to avoid permanent global mutation")
        
        self.noise_schedule = noise_schedule or CosineSchedule(device=self.device, dtype=self.dtype)

        # Avoid unnecessary device transfer if already on target device/dtype
        try:
            # Cache first parameter to avoid consuming iterator twice
            first_param = next(score_model.parameters())
            current_device = first_param.device
            current_dtype = first_param.dtype

            if current_device != self.device or current_dtype != self.dtype:
                logger.debug(f"Moving score model from {current_device}:{current_dtype} to {self.device}:{self.dtype}")
                self.score_model = score_model.to(self.device, self.dtype)
            else:
                logger.debug(f"Score model already on {self.device}:{self.dtype}, skipping transfer")
                self.score_model = score_model
        except StopIteration:
            # Model has no parameters - still need to transfer (for buffers)
            logger.debug("Score model has no parameters, transferring anyway (for buffers)")
            self.score_model = score_model.to(self.device, self.dtype)

        self._apply_memory_format_to_module(self.score_model)

        self._model_outputs_score = self._infer_model_outputs_score(self.score_model)
        # surface attributes to downstream code
        setattr(self.score_model, "predicts_score", self._model_outputs_score)
        setattr(self.score_model, "predicts_noise", not self._model_outputs_score)

        self._compiled = False
        self._compile_warmup_done = not self.config.compile_score_model or not self.config.compile_warmup
        if self.config.compile_score_model:
            if hasattr(torch, "compile"):
                try:
                    self.score_model = torch.compile(
                        self.score_model,
                        mode=self.config.compile_mode,
                        fullgraph=self.config.compile_fullgraph,
                    )
                    self._compiled = True
                    if not self.config.compile_warmup:
                        self._compile_warmup_done = True
                    logger.info("Enabled torch.compile for score model")
                except Exception as exc:
                    logger.warning(f"torch.compile failed: {exc}; continuing without compilation")
                    self._compiled = False
                    self._compile_warmup_done = True
            else:
                logger.debug("torch.compile not available in this runtime; skipping compilation request")
                self._compile_warmup_done = True

        # Validate score model signature
        try:
            with torch.no_grad():
                test_x = torch.randn(1, 3, 8, 8, device=self.device, dtype=self.dtype)
                if self.config.timestep_shape_b1:
                    test_t = torch.full((1, 1), 0.5, device=self.device, dtype=torch.float32)
                else:
                    test_t = torch.full((1,), 0.5, device=self.device, dtype=torch.float32)
                test_out = self.score_model(test_x, test_t)
                if test_out.shape != test_x.shape:
                    raise ValueError(f"Score model output shape {test_out.shape} doesn't match input {test_x.shape}")
        except Exception as e:
            logger.error(f"Score model validation failed: {e}")
            raise ValueError(f"Score model failed validation: {e}. Model must accept (x, t) and return tensor of same shape as x")
        
        # Production release: Document this behavior
        try:
            self.score_model.eval()
            for p in self.score_model.parameters():
                p.requires_grad_(False)
            logger.debug("Score model set to eval mode with frozen parameters")
        except Exception as e:
            logger.warning(f"Could not set model to eval mode or freeze parameters: {e}")
        
        self.sinkhorn_kernel = OptimizedSinkhornKernel(self.device, self.dtype, self.config)
        self.dpm_solver = DPMSolverPP(self.config.dpm_solver_order, schedule=self.noise_schedule)

        self._patch_transport = None
        self._active_patch_size = self.config.patch_size
        self._active_blockwise_threshold = self.config.blockwise_threshold
        if self.device.type == 'cpu':
            cpu_threshold = min(64_000_000, self.config.max_dense_matrix_elements)
            if self._active_blockwise_threshold < cpu_threshold:
                logger.debug(
                    "Raising blockwise threshold for CPU to %d (was %d)",
                    cpu_threshold,
                    self._active_blockwise_threshold,
                )
                self._active_blockwise_threshold = int(cpu_threshold)
                self.config.blockwise_threshold = int(cpu_threshold)
        self._ensure_patch_transport(self._active_patch_size)

        # Initialize alternative integrators if requested
        self.integrator = None
        if INTEGRATORS_AVAILABLE and self.config.integrator != "dpm_solver++":
            if self.config.integrator == "heun":
                self.integrator = HeunIntegrator(self.noise_schedule)
                logger.debug("Using Heun integrator (2nd order Runge-Kutta)")
            elif self.config.integrator == "ddim":
                self.integrator = DDIMIntegrator(self.noise_schedule, eta=self.config.ddim_eta)
                logger.debug(f"Using DDIM integrator (eta={self.config.ddim_eta})")
            elif self.config.integrator == "adaptive":
                self.integrator = AdaptiveIntegrator(
                    self.noise_schedule,
                    atol=self.config.adaptive_atol,
                    rtol=self.config.adaptive_rtol
                )
                logger.debug(f"Using adaptive integrator (atol={self.config.adaptive_atol}, rtol={self.config.adaptive_rtol})")
            elif self.config.integrator == "exponential":
                self.integrator = ExponentialIntegrator(self.noise_schedule)
                logger.debug("Using exponential integrator")

        # Initialize corrector if requested
        self.corrector = None
        if CORRECTORS_AVAILABLE and self.config.use_corrector:
            if self.config.corrector_type == "langevin":
                self.corrector = LangevinCorrector(
                    self.noise_schedule,
                    n_steps=self.config.corrector_steps,
                    snr=self.config.langevin_snr
                )
                logger.debug(f"Using Langevin corrector (steps={self.config.corrector_steps}, snr={self.config.langevin_snr})")
            elif self.config.corrector_type == "tweedie":
                self.corrector = TweedieCorrector(
                    self.noise_schedule,
                    mixing=self.config.tweedie_mixing
                )
                logger.debug(f"Using Tweedie corrector (mixing={self.config.tweedie_mixing})")
            elif self.config.corrector_type == "adaptive":
                self.corrector = AdaptiveCorrector(
                    self.noise_schedule,
                    error_threshold=self.config.adaptive_corrector_threshold,
                    langevin_snr=self.config.langevin_snr
                )
                logger.debug(f"Using adaptive corrector (threshold={self.config.adaptive_corrector_threshold})")

        if self.config.deterministic and self.sinkhorn_kernel.use_pot:
            logger.debug("(Deterministic mode) POT backend disabled to ensure reproducibility")
        
        self._state_lock = threading.RLock()
        self._thread_local = threading.local()
        
        if self.config.use_patch_based_ot:
            self._patch_transport = make_grid_patch_transport(self, self.config.patch_size)
        else:
            self._patch_transport = None
        
        self.timing_stats = {
            'score_eval': deque(maxlen=100),
            'integration': deque(maxlen=100),
            'total_step': deque(maxlen=100),
            'richardson_overhead': deque(maxlen=100)
        }
        
        self._det_ctx_uses = 0
        self._det_ctx_steps = 0
        
        self.fallback_count = 0
        self.memory_stats = deque(maxlen=200)

        logger.info(f"SPOT {__version__} initialized (compute_dtype={self.dtype})")

    def _increment_fallback(self):
        """Thread-safe fallback counter increment."""
        with self._state_lock:
            self.fallback_count += 1
    
    def _enable_tf32_internal(self):
        """Internal method to enable TF32 settings."""
        try:
            self._original_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
            self._original_tf32_cudnn = torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            if hasattr(torch, 'get_float32_matmul_precision'):
                self._original_matmul_prec = torch.get_float32_matmul_precision()
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
            
            self._tf32_enabled = True
            logger.debug("Enabled TF32 for matmul operations (explicit opt-in)")
        except Exception as e:
            logger.warning(f"Could not enable TF32: {e}")
    
    def _restore_tf32_settings(self):
        """Restore original TF32 settings."""
        if not self._tf32_enabled:
            return
        
        try:
            if self._original_tf32_matmul is not None:
                torch.backends.cuda.matmul.allow_tf32 = self._original_tf32_matmul
                logger.debug("Restored original TF32 matmul setting")
        except Exception as e:
            logger.warning(f"Could not restore TF32 matmul setting: {e}")
        
        try:
            if self._original_tf32_cudnn is not None:
                torch.backends.cudnn.allow_tf32 = self._original_tf32_cudnn
                logger.debug("Restored original TF32 cudnn setting")
        except Exception as e:
            logger.warning(f"Could not restore TF32 cudnn setting: {e}")
        
        try:
            if self._original_matmul_prec is not None and hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision(self._original_matmul_prec)
                logger.debug("Restored original matmul precision")
        except Exception as e:
            logger.warning(f"Could not restore matmul precision: {e}")
        
        self._tf32_enabled = False
    
    @contextmanager
    def tf32_context(self):
        """Context manager for safe TF32 usage."""
        if not torch.cuda.is_available():
            yield
            return
            
        original_matmul = torch.backends.cuda.matmul.allow_tf32
        original_cudnn = torch.backends.cudnn.allow_tf32
        original_prec = None
        if hasattr(torch, 'get_float32_matmul_precision'):
            original_prec = torch.get_float32_matmul_precision()
        
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original_matmul
            torch.backends.cudnn.allow_tf32 = original_cudnn
            if original_prec is not None and hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision(original_prec)
    
    @contextmanager
    def _deterministic_context(self):
        """Context manager for deterministic mode without permanent global state mutation."""
        if not self.config.deterministic:
            yield
            return
        
        prev_benchmark = None
        prev_deterministic = None
        if torch.cuda.is_available():
            prev_benchmark = torch.backends.cudnn.benchmark
            prev_deterministic = torch.backends.cudnn.deterministic
        
        prev_algos = None
        try:
            prev_algos = torch.are_deterministic_algorithms_enabled()
        except (AttributeError, RuntimeError):
            prev_algos = None
        
        try:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                logger.debug("Enabled fully deterministic mode (may impact performance)")
                if self.device.type == 'cuda' and not self.config.deterministic_cdist_cpu:
                    logger.debug("Note: Determinism may be limited by CUDA cdist on some drivers/kernels")
                    logger.debug("Enable deterministic_cdist_cpu=True for bit-exact determinism (slower)")
                elif self.config.deterministic_cdist_cpu:
                    logger.debug("Using CPU for cdist operations (bit-exact determinism, reduced performance)")
            except Exception as e:
                logger.debug(f"Could not enable deterministic algorithms: {e}")
                if torch.cuda.is_available():
                    logger.debug("Deterministic mode partially enabled (cudnn only)")
            
            self._det_ctx_uses += 1
            
            yield
            
        finally:
            try:
                if torch.cuda.is_available() and prev_benchmark is not None:
                    torch.backends.cudnn.benchmark = prev_benchmark
                    torch.backends.cudnn.deterministic = prev_deterministic
                if prev_algos is not None and hasattr(torch, 'use_deterministic_algorithms'):
                    torch.use_deterministic_algorithms(bool(prev_algos), warn_only=True)
                elif prev_algos is None and hasattr(torch, 'use_deterministic_algorithms'):
                    torch.use_deterministic_algorithms(False, warn_only=True)
            except Exception as e:
                logger.debug(f"Could not fully restore deterministic settings: {e}")
    
    @contextmanager
    def memory_profiler(self):
        """Context manager for memory profiling."""
        if not self.config.profile_memory or not torch.cuda.is_available():
            yield
            return

        start_mem = torch.cuda.memory_allocated()
        try:
            yield
        finally:
            end_mem = torch.cuda.memory_allocated()
            self.memory_stats.append(end_mem - start_mem)

    def _infer_model_outputs_score(self, model: torch.nn.Module) -> bool:
        if hasattr(model, "predicts_noise"):
            return not bool(getattr(model, "predicts_noise"))
        if hasattr(model, "predicts_score"):
            return bool(getattr(model, "predicts_score"))
        return True

    def _convert_model_output_to_score(self, model_output: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        if self._model_outputs_score:
            return model_output

        alpha, sigma = self.noise_schedule.alpha_sigma(t_tensor)
        sigma = sigma.to(model_output.dtype)

        while sigma.ndim < model_output.ndim:
            sigma = sigma.unsqueeze(-1)

        return -model_output / torch.clamp(sigma, min=EPSILON_CLAMP)

    def _apply_memory_format_to_module(self, module: torch.nn.Module) -> None:
        if self._memory_format != torch.channels_last:
            return

        try:
            module.to(memory_format=self._memory_format)
            for param in module.parameters(recurse=True):
                if param.dim() == 4:
                    param.data = param.data.contiguous(memory_format=self._memory_format)
            for buffer in module.buffers():
                if buffer.dim() == 4:
                    buffer.data = buffer.data.contiguous(memory_format=self._memory_format)
            logger.debug("Applied channels_last memory format to score model")
        except Exception as exc:
            logger.debug(f"Could not apply channels_last memory format: {exc}")

    def _format_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._memory_format == torch.channels_last and tensor.dim() == 4:
            return tensor.contiguous(memory_format=torch.channels_last)
        return tensor

    def _ensure_patch_transport(self, patch_size: int) -> None:
        if not self.config.use_patch_based_ot:
            self._patch_transport = None
            return

        self.config.patch_size = int(patch_size)
        self._patch_transport = make_grid_patch_transport(self, int(patch_size))

    def _auto_tune_for_shape(self, shape: Union[torch.Size, Tuple[int, ...]]) -> None:
        if not self.config.auto_tune_highres:
            return
        if len(shape) != 4:
            return

        _, _, height, width = shape
        largest_dim = max(height, width)

        # Always start from the base configuration so that subsequent calls can
        # both increase *and* decrease thresholds depending on the requested
        # resolution.  The previous implementation initialised from the
        # currently active values and then only took the ``max`` with the
        # high/ultra-res thresholds.  On CPU we inflate the default threshold to
        # ~50M elements which meant the automatic tuning logic could never lower
        # it to the user-specified high/ultra-res thresholds.  The tests expect
        # those explicit configuration values to win, so derive the targets from
        # the configuration each time we auto-tune.
        new_patch = self.config.patch_size
        new_threshold = self.config.blockwise_threshold

        if largest_dim >= 2048:
            new_patch = max(new_patch, self.config.ultrares_patch_size)
            new_threshold = self.config.ultrares_blockwise_threshold
        elif largest_dim >= 1024:
            new_patch = max(new_patch, self.config.highres_patch_size)
            new_threshold = self.config.highres_blockwise_threshold

        if self.device.type == "cpu":
            cpu_limit = min(64_000_000, self.config.max_dense_matrix_elements)
            new_threshold = min(new_threshold, cpu_limit)

        if new_patch != self._active_patch_size:
            logger.info(
                "Auto-tuning patch transport for %dx%d -> patch_size=%d",
                height,
                width,
                new_patch,
            )
            self._active_patch_size = int(new_patch)
            self._ensure_patch_transport(self._active_patch_size)

        if new_threshold != self._active_blockwise_threshold:
            logger.debug(
                "Auto-tuning blockwise threshold for %dx%d -> %d",
                height,
                width,
                new_threshold,
            )
            self._active_blockwise_threshold = int(new_threshold)
            self.config.blockwise_threshold = int(new_threshold)

    def _maybe_warmup(self, shape: Union[torch.Size, Tuple[int, ...]]) -> None:
        if not self._compiled or self._compile_warmup_done:
            return

        try:
            warmup_shape = tuple(shape)
            dummy = torch.zeros(warmup_shape, device=self.device, dtype=self.dtype)
            dummy = self._format_tensor(dummy)
            if self.config.timestep_shape_b1:
                t = torch.zeros((warmup_shape[0], 1), device=self.device, dtype=torch.float32)
            else:
                t = torch.zeros((warmup_shape[0],), device=self.device, dtype=torch.float32)
            with torch.inference_mode():
                self.score_model(dummy, t)
            logger.debug("Completed torch.compile warmup run")
        except Exception as exc:
            logger.warning(f"Warmup invocation failed: {exc}")
        finally:
            self._compile_warmup_done = True

    def _compute_score_optimized(self, x, t):
        """Compute score with gradient safety checks."""
        if x.numel() > self.config.max_tensor_size_elements:
            raise ValueError(f"Input tensor too large: {x.numel()}")
        
        if x.requires_grad:
            raise RuntimeError("Score computation requires inference mode (x.requires_grad must be False)")
        
        if self.config.timestep_shape_b1:
            t_tensor = torch.full((x.size(0), 1), float(t), device=x.device, dtype=torch.float32)
        else:
            t_tensor = torch.full((x.size(0),), float(t), device=x.device, dtype=torch.float32)
        
        if t_tensor.requires_grad:
            raise RuntimeError("Score computation requires inference mode (t.requires_grad must be False)")
        
        x_in = self._format_tensor(x)

        with torch.amp.autocast(**self._autocast_kwargs):
            model_out = self.score_model(x_in, t_tensor)

        score = self._convert_model_output_to_score(model_out, t_tensor)
        return score.to(x.dtype)
    
    def _should_use_richardson(self, score):
        """Determine if Richardson extrapolation should be used."""
        if not hasattr(self._thread_local, 'richardson_state'):
            self._thread_local.richardson_state = {
                'prev_score': None,
                'score_change_history': deque(maxlen=10),
                'last_shape': None
            }
        
        state = self._thread_local.richardson_state
        
        current_shape = tuple(score.shape)
        if state['last_shape'] != current_shape:
            state['prev_score'] = None
            state['score_change_history'].clear()
            state['last_shape'] = current_shape
            return False
        
        if not self.config.richardson_extrapolation:
            return False
        
        if state['prev_score'] is None:
            state['prev_score'] = score.clone().detach()
            return False
        
        score_diff = score - state['prev_score']
        score_change = torch.norm(score_diff.float()).item()
        score_norm = torch.norm(score.float()).item()
        normalized_change = score_change / (score_norm + EPSILON_CLAMP) / math.sqrt(score.numel())
        
        state['score_change_history'].append(normalized_change)
        state['prev_score'] = score.clone().detach()
        
        if len(state['score_change_history']) >= 3:
            recent_change = np.mean(list(state['score_change_history'])[-3:])
            return recent_change > self.config.richardson_threshold
        
        return False
    
    def _suggest_eps(self, X, Y, base):
        """Data-scaled epsilon suggestion."""
        with torch.no_grad():
            sample = min(2048, X.shape[0], Y.shape[0])
            Xi = X[:sample].float()
            Yi = Y[:sample].float()
            C = torch.cdist(Xi, Yi, p=2).pow(2)
            med = torch.median(C).item()
        return max(EPSILON_MIN, base * (med + 1e-12))
    
    def _per_sample_pixel_transport(self, xb, yb, eps):
        """Per-pixel transport treating pixels as points in color space."""
        if eps < EPSILON_MIN:
            logger.debug(f"Epsilon too small ({eps:.2e}) in per-pixel transport, using identity")
            return lambda z: z
        
        _, C, H, W = xb.shape
        
        X = xb.permute(0, 2, 3, 1).reshape(-1, C)
        Y = yb.permute(0, 2, 3, 1).reshape(-1, C)
        
        if self.config.adaptive_eps_scale == 'data':
            eps = self._suggest_eps(X, Y, eps)
        
        n_pixels = H * W
        if xb.device.type == 'cpu' and n_pixels * n_pixels > 16_000_000:
            logger.debug(
                "CPU fallback: per-pixel OT (%d^2 elements) too large, using identity",
                n_pixels,
            )
            self._increment_fallback()
            return lambda z: z
        if xb.is_cuda:
            bytes_per_elem = 4
            total_mem = torch.cuda.get_device_properties(xb.device).total_memory
            max_elems = min(
                int(total_mem * DEFAULT_GPU_MEMORY_FRACTION / bytes_per_elem),
                self.config.max_dense_matrix_elements
            )
        else:
            # Allow substantially larger dense problems on CPU now that the solver
            # uses more efficient tensor reuse. The previous 2M cap caused the
            # common 64x64 case to fall back to the extremely slow blockwise
            # transport, so we raise the heuristic limit to 64M elements
            # (~256MB for fp32) which remains conservative for CI environments
            # and still respects the config-level cap.
            max_elems = min(64_000_000, self.config.max_dense_matrix_elements)
        
        needs_blockwise = (n_pixels * n_pixels > max_elems) or (
            n_pixels * n_pixels > self._active_blockwise_threshold
        )
        
        log_u, log_v = self.sinkhorn_kernel.sinkhorn_log_stabilized(
            X, Y, eps, n_iter=self.config.sinkhorn_iterations
        )

        if not (torch.isfinite(log_u).all() and torch.isfinite(log_v).all()):
            logger.debug(f"Per-pixel Sinkhorn failed, using identity transport")
            self._increment_fallback()
            if self.fallback_count > self.config.max_fallbacks:
                logger.warning(f"Excessive fallbacks ({self.fallback_count}), check configuration")
            return lambda z: z

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda' if X.is_cuda else 'cpu', enabled=False):
                if not needs_blockwise:
                    if self.config.deterministic_cdist_cpu and X.is_cuda:
                        X_cpu = X.float().cpu()
                        Y_cpu = Y.float().cpu()
                        C_xy = torch.cdist(X_cpu, Y_cpu, p=2).pow(2).to(X.device)
                    else:
                        C_xy = torch.cdist(X.float(), Y.float(), p=2).pow(2)
                    S = -C_xy / eps
                    log_weights = log_u.float()[:, None] + S + log_v.float()[None, :]
                    row_max = log_weights.max(dim=1, keepdim=True).values
                    weights = torch.exp(log_weights - row_max)
                    rows = weights.sum(1, keepdim=True).clamp_min(ROW_SUM_MIN)
                    Ybar = (weights @ Y.float()) / rows
                else:
                    Ybar = torch.zeros_like(X, dtype=torch.float32)
                    log_u32, log_v32 = log_u.float(), log_v.float()

                    memory_ratio = min(1.0, max_elems / 50_000_000)
                    optimal_block = int(MAX_BLOCK_SIZE * (1 + memory_ratio))
                    block_size = max(MIN_BLOCK_SIZE, min(optimal_block, int(max_elems // max(1, n_pixels))))

                    Y_cpu = Y.float().cpu() if (self.config.deterministic_cdist_cpu and X.is_cuda) else None

                    row_max = torch.full((n_pixels, 1), -torch.inf, device=X.device, dtype=torch.float32)
                    row_sum = torch.zeros((n_pixels, 1), device=X.device, dtype=torch.float32)

                    for i in range(0, n_pixels, block_size):
                        end_i = min(i + block_size, n_pixels)
                        X_block = X[i:end_i].float()

                        if Y_cpu is not None:
                            X_block_cpu = X_block.cpu()
                            C_block = torch.cdist(X_block_cpu, Y_cpu, p=2).pow(2).to(X.device)
                        else:
                            C_block = torch.cdist(X_block, Y.float(), p=2).pow(2)
                        
                        S_block = -C_block / eps
                        log_weights = log_u32[i:end_i, None] + S_block + log_v32[None, :]
                        block_max = log_weights.max(dim=1, keepdim=True).values
                        weights = torch.exp(log_weights - block_max)
                        block_sum = weights.sum(dim=1, keepdim=True)
                        block_num = weights @ Y.float()

                        new_max = torch.maximum(row_max[i:end_i], block_max)
                        existing_scale = torch.exp(row_max[i:end_i] - new_max)
                        block_scale = torch.exp(block_max - new_max)

                        row_sum[i:end_i] = row_sum[i:end_i] * existing_scale + block_sum * block_scale
                        Ybar[i:end_i] = Ybar[i:end_i] * existing_scale + block_num * block_scale
                        row_max[i:end_i] = new_max

                    rows = row_sum.clamp_min(ROW_SUM_MIN)
                    Ybar = Ybar / rows

        def transport_map(zb):
            Z = zb.permute(0, 2, 3, 1).reshape(-1, C)

            if needs_blockwise:
                memory_ratio = min(1.0, max_elems / 50_000_000)
                optimal_block = int(MAX_BLOCK_SIZE * (1 + memory_ratio))
                bs = max(MIN_BLOCK_SIZE, min(optimal_block, int(max_elems // max(1, n_pixels))))
                Z_next = blockwise_soft_assignment(Z, X, eps, Ybar, block_size=bs, 
                                                   deterministic_cdist_cpu=self.config.deterministic_cdist_cpu)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda' if Z.is_cuda else 'cpu', enabled=False):
                        if self.config.deterministic_cdist_cpu and Z.is_cuda:
                            Z_cpu = Z.float().cpu()
                            X_cpu = X.float().cpu()
                            C_zx = torch.cdist(Z_cpu, X_cpu, p=2).pow(2).to(Z.device)
                        else:
                            C_zx = torch.cdist(Z.float(), X.float(), p=2).pow(2)
                        S = -C_zx / eps
                        S = S - S.max(dim=1, keepdim=True)[0]
                        Z_next = (torch.softmax(S, dim=1) @ Ybar).to(Z.dtype)
            
            return Z_next.view(1, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        return transport_map
    
    def _standard_transport(self, x, y, eps):
        """Standard transport with per-pixel OT."""
        B = x.size(0)
        
        if B == 1 and self._patch_transport is not None and not self.config.force_per_pixel_b1:
            return self._patch_transport(x, y, eps)
        
        transport_maps = []
        for b in range(B):
            xb = x[b:b+1]
            yb = y[b:b+1]
            transport_maps.append(self._per_sample_pixel_transport(xb, yb, eps))
        
        def batched_transport(z):
            outputs = []
            for b, tm in enumerate(transport_maps):
                outputs.append(tm(z[b:b+1]))
            return torch.cat(outputs, dim=0)
        
        return batched_transport
    
    def _standard_transport_fallback(self, x, y, eps):
        """Fallback for batch > 1 in patch transport."""
        B = x.size(0)
        
        image_pairs = [(x[b:b+1], y[b:b+1]) for b in range(B)]
        cached_transports = [None] * B
        
        def batched_transport(z):
            outputs = []
            for b in range(B):
                if cached_transports[b] is None:
                    xb_img, yb_img = image_pairs[b]

                    if z.dim() == 4:
                        cached_transports[b] = self._per_sample_pixel_transport(xb_img, yb_img, eps)
                    else:
                        xb_flat = xb_img.reshape(xb_img.size(0), -1)
                        yb_flat = yb_img.reshape(yb_img.size(0), -1)

                        if xb_flat.size(0) == 1 and yb_flat.size(0) == 1:
                            logger.warning("N=M=1 degenerate case, using identity fallback - may affect quality")
                            self._increment_fallback()
                            cached_transports[b] = lambda z: z
                        else:
                            log_u, log_v = self.sinkhorn_kernel.sinkhorn_log_stabilized(
                                xb_flat, yb_flat, eps,
                                n_iter=self.config.sinkhorn_iterations
                            )
                            if not (torch.isfinite(log_u).all() and torch.isfinite(log_v).all()):
                                self._increment_fallback()
                                cached_transports[b] = lambda z: z
                            else:
                                cached_transports[b] = self._create_transport_map(xb_flat, yb_flat, log_u, log_v, eps)
                
                outputs.append(cached_transports[b](z[b:b+1]))
            
            return torch.cat(outputs, dim=0)
        
        return batched_transport
    
    def _create_transport_map(self, x, y, log_u, log_v, eps, allow_single_point=False):
        """Create transport map from dual variables with blockwise computation for large problems.

        Args:
            x: Source points
            y: Target points
            log_u, log_v: Sinkhorn dual variables in log-space
            eps: Entropic regularization
            allow_single_point: If True, allows N=M=1 case (e.g., for single-patch transport)
        """
        if not (torch.isfinite(log_u).all() and torch.isfinite(log_v).all()):
            logger.debug("Non-finite duals, returning identity transport")
            self._increment_fallback()
            return lambda z: z

        if eps < EPSILON_MIN:
            logger.debug(f"Epsilon too small ({eps:.2e}), returning identity")
            self._increment_fallback()
            return lambda z: z

        xf = x.reshape(x.size(0), -1)
        yf = y.reshape(y.size(0), -1)

        n, m = xf.size(0), yf.size(0)

        if n == 1 and m == 1 and not allow_single_point:
            logger.warning(
                "N=M=1 degenerate case detected. Using identity transport. "
                "This typically indicates invalid input data or configuration."
            )
            self._increment_fallback()
            # Use pure identity for degenerate N=M=1 case
            def identity_transport(z):
                return z
            return identity_transport
        
        if xf.is_cuda:
            bytes_per_elem = 4
            total_mem = torch.cuda.get_device_properties(xf.device).total_memory
            max_elems = min(
                int(total_mem * DEFAULT_GPU_MEMORY_FRACTION / bytes_per_elem),
                self.config.max_dense_matrix_elements
            )
        else:
            max_elems = min(64_000_000, self.config.max_dense_matrix_elements)
        
        required_elems = n * m
        if xf.device.type == 'cpu' and required_elems > 16_000_000:
            logger.debug(
                "CPU fallback: transport %dx%d=%d elements too large, using identity",
                n,
                m,
                required_elems,
            )
            self._increment_fallback()
            return lambda z: z

        if required_elems > max_elems * 2:
            logger.debug(f"Transport problem too large: {n}x{m}={required_elems} elements, using identity")
            self._increment_fallback()
            return lambda z: z
        
        needs_blockwise_hw = (required_elems > max_elems)
        needs_blockwise_cfg = (required_elems > self._active_blockwise_threshold)
        needs_blockwise = needs_blockwise_hw or needs_blockwise_cfg
        
        if needs_blockwise:
            logger.debug(f"Using blockwise transport for {n}x{m} problem (max_elems={max_elems})")
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda' if xf.is_cuda else 'cpu', enabled=False):
                if not needs_blockwise:
                    if self.config.deterministic_cdist_cpu and xf.is_cuda:
                        xf_cpu = xf.float().cpu()
                        yf_cpu = yf.float().cpu()
                        C_xy = torch.cdist(xf_cpu, yf_cpu, p=2).pow(2).to(xf.device)
                    else:
                        C_xy = torch.cdist(xf.float(), yf.float(), p=2).pow(2)
                    S = -C_xy / eps
                    log_weights = log_u.float()[:, None] + S + log_v.float()[None, :]
                    row_max = log_weights.max(dim=1, keepdim=True).values
                    weights = torch.exp(log_weights - row_max)
                    row_sum = weights.sum(1, keepdim=True).clamp_min(ROW_SUM_MIN)
                    Ybar = (weights @ yf.float()) / row_sum
                else:
                    Ybar = torch.zeros_like(xf, dtype=torch.float32)
                    row_sum = torch.zeros(n, 1, device=xf.device, dtype=torch.float32)
                    log_u32, log_v32 = log_u.float(), log_v.float()
                    row_max = torch.full((n, 1), -torch.inf, device=xf.device, dtype=torch.float32)

                    memory_ratio = min(1.0, max_elems / 50_000_000)
                    optimal_block = int(MAX_BLOCK_SIZE * (1 + memory_ratio))
                    block_size = max(MIN_BLOCK_SIZE, min(optimal_block, int(max_elems // max(1, m))))

                    yf_cpu = yf.float().cpu() if (self.config.deterministic_cdist_cpu and xf.is_cuda) else None

                    for i in range(0, n, block_size):
                        end_i = min(i + block_size, n)
                        xf_block = xf[i:end_i].float()

                        if yf_cpu is not None:
                            xf_block_cpu = xf_block.cpu()
                            C_block = torch.cdist(xf_block_cpu, yf_cpu, p=2).pow(2).to(xf.device)
                        else:
                            C_block = torch.cdist(xf_block, yf.float(), p=2).pow(2)
                        S_block = -C_block / eps
                        log_weights = log_u32[i:end_i, None] + S_block + log_v32[None, :]
                        block_max = log_weights.max(dim=1, keepdim=True).values
                        weights = torch.exp(log_weights - block_max)
                        block_sum = weights.sum(dim=1, keepdim=True)
                        block_num = weights @ yf.float()

                        new_max = torch.maximum(row_max[i:end_i], block_max)
                        existing_scale = torch.exp(row_max[i:end_i] - new_max)
                        block_scale = torch.exp(block_max - new_max)

                        row_sum[i:end_i] = row_sum[i:end_i] * existing_scale + block_sum * block_scale
                        Ybar[i:end_i] = Ybar[i:end_i] * existing_scale + block_num * block_scale
                        row_max[i:end_i] = new_max

                    row_sum = row_sum.clamp_min(ROW_SUM_MIN)
                    Ybar = Ybar / row_sum
        
        def transport_map(z):
            try:
                zf = z.reshape(z.size(0), -1)
                
                if needs_blockwise:
                    memory_ratio = min(1.0, max_elems / 50_000_000)
                    optimal_block = int(MAX_BLOCK_SIZE * (1 + memory_ratio))
                    bs = max(MIN_BLOCK_SIZE, min(optimal_block, int(max_elems // max(1, m))))
                    z_next_flat = blockwise_soft_assignment(
                        zf, xf, eps, Ybar, 
                        block_size=bs,
                        deterministic_cdist_cpu=self.config.deterministic_cdist_cpu
                    )
                else:
                    with torch.no_grad():
                        with torch.amp.autocast(device_type='cuda' if zf.is_cuda else 'cpu', enabled=False):
                            if self.config.deterministic_cdist_cpu and zf.is_cuda:
                                zf_cpu = zf.float().cpu()
                                xf_cpu = xf.float().cpu()
                                C_zx = torch.cdist(zf_cpu, xf_cpu, p=2).pow(2).to(zf.device)
                            else:
                                C_zx = torch.cdist(zf.float(), xf.float(), p=2).pow(2)
                            S = -C_zx / eps
                            S = S - S.max(dim=1, keepdim=True)[0]
                            z_next_flat = (torch.softmax(S, dim=1) @ Ybar).to(zf.dtype)
                
                return z_next_flat.reshape(z.shape)
                
            except Exception as e:
                logger.warning(
                    f"Transport map application failed (fallback #{self.fallback_count + 1}): {e}. "
                    f"Using identity transform. Exception type: {type(e).__name__}"
                )
                # Production release: Increment fallback_count in exception handler
                self._increment_fallback()
                if self.fallback_count > self.config.max_fallbacks:
                    logger.error(
                        f"Excessive transport fallbacks ({self.fallback_count} > {self.config.max_fallbacks}). "
                        f"This may indicate a serious configuration or numerical stability issue."
                    )
                return z
        
        return transport_map
    
    @torch.inference_mode()
    def sample_enhanced(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 12,
        verbose: bool = True,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        return_stats: bool = False,
        output_dtype: Optional[torch.dtype] = torch.float32,
        use_result_dataclass: bool = False
    ):
        """Main sampling function with all fixes applied."""

        # Input validation for better error messages
        if not isinstance(shape, (tuple, list)) or len(shape) < 3:
            raise ValueError(f"Shape must be tuple/list with >=3 dimensions, got: {shape}")

        if num_steps < 2:
            raise ValueError(f"num_steps must be at least 2, got: {num_steps}")

        if any(dim <= 0 for dim in shape):
            raise ValueError(f"All dimensions must be positive, got shape: {shape}")

        shape = tuple(int(dim) for dim in shape)

        total_elements = np.prod(shape)
        if total_elements > self.config.max_tensor_size_elements:
            raise ValueError(
                f"Requested shape {shape} has {total_elements} elements, "
                f"exceeds max_tensor_size_elements={self.config.max_tensor_size_elements}"
            )
        
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        self._auto_tune_for_shape(shape)
        self._maybe_warmup(shape)

        # Production release: Thread-safe Richardson flag (no config mutation)
        use_richardson = self.config.richardson_extrapolation and not self.config.deterministic
        
        tf32_cm = self.tf32_context() if (self.config.enable_tf32 and torch.cuda.is_available()) else nullcontext()

        try:
            with tf32_cm:
                with self._deterministic_context():
                    if self.config.deterministic and torch.cuda.is_available():
                        assert torch.backends.cudnn.deterministic, "Deterministic context not active"
                    
                    if not hasattr(self._thread_local, 'richardson_state'):
                        self._thread_local.richardson_state = {
                            'prev_score': None,
                            'score_change_history': deque(maxlen=10),
                            'last_shape': None
                        }
                    else:
                        self._thread_local.richardson_state['prev_score'] = None
                        self._thread_local.richardson_state['score_change_history'].clear()
                        self._thread_local.richardson_state['last_shape'] = None
                    
                    timesteps = self.dpm_solver.get_timesteps(num_steps, device=torch.device('cpu'))
                    
                    if generator is not None:
                        x_t = torch.randn(shape, device=self.device, dtype=self.dtype, generator=generator)
                    else:
                        x_t = torch.randn(shape, device=self.device, dtype=self.dtype)
                    x_t = self._format_tensor(x_t)
                    
                    model_outputs = deque(maxlen=self.dpm_solver.order)
                    
                    iterator = tqdm(range(len(timesteps) - 1), desc=f"SPOT {__version__}") if verbose else range(len(timesteps) - 1)
                    
                    for i in iterator:
                        if self.config.deterministic:
                            ctx_active = False
                            if torch.cuda.is_available() and torch.backends.cudnn.deterministic:
                                ctx_active = True
                            else:
                                try:
                                    ctx_active = torch.are_deterministic_algorithms_enabled()
                                except (AttributeError, RuntimeError):
                                    pass
                            if ctx_active:
                                self._det_ctx_steps += 1
                        
                        step_start = time.time()
                        
                        t_curr, t_next = timesteps[i], timesteps[i + 1]
                        dt = abs(t_next - t_curr)
                        
                        self._thread_local.step_context = StepContext(current_t=t_curr, current_dt=dt)
                        
                        score_start = time.time()
                        with self.memory_profiler():
                            score = self._compute_score_optimized(x_t, t_curr)
                        model_outputs.append(score)
                        self.timing_stats['score_eval'].append(time.time() - score_start)
                        
                        # Production release: Avoid redundant t_tensor computation
                        def drift_fn(x, t):
                            t_tensor = torch.full((1,), float(t), device=x.device, dtype=torch.float32)

                            if abs(t - t_curr) < 1e-8:
                                s = score
                            else:
                                s = self._compute_score_optimized(x, t)

                            _, sigma_t = self.noise_schedule.alpha_sigma(t_tensor)
                            beta_t = self.noise_schedule.beta(t_tensor).to(x.dtype)
                            
                            with torch.amp.autocast(device_type='cuda' if x.is_cuda else 'cpu', enabled=False):
                                sigma_sq = (sigma_t.to(x.dtype)) ** 2
                            # Probability-flow ODE drift to match HeunIntegrator
                            return -0.5 * beta_t * x - beta_t * sigma_sq * s
                        
                        def compute_eps_at(t_scalar):
                            eps_base = self.config.eps
                            if self.config.adaptive_eps and self.config.adaptive_eps_scale == 'sigma':
                                t_tensor = torch.full((1,), float(t_scalar), device=self.device, dtype=torch.float32)
                                _, sigma_curr = self.noise_schedule.alpha_sigma(t_tensor)
                                return max(EPSILON_MIN, eps_base * (float(sigma_curr.float().item()) + EPSILON_CLAMP))
                            return max(EPSILON_MIN, eps_base)
                        
                        def _dpm_solver_predict(x_current):
                            return self.dpm_solver.multistep_update(x_current, model_outputs, timesteps, i)

                        integration_start = time.time()
                        with self.memory_profiler():
                            # Choose integrator
                            if self.integrator is not None:
                                # Use alternative integrator
                                def score_fn_wrapper(x_val, t_val):
                                    return self._compute_score_optimized(x_val, t_val)

                                # Different integrators have different interfaces
                                if isinstance(self.integrator, DDIMIntegrator):
                                    # DDIM needs generator for optional stochasticity
                                    y_pred = self.integrator.step(x_t, t_curr, t_next, score_fn_wrapper, generator)
                                elif hasattr(self.integrator, 'step'):
                                    # All other integrators (Heun, Exponential, Adaptive) use standard step interface
                                    y_pred = self.integrator.step(x_t, t_curr, t_next, score_fn_wrapper)
                                else:
                                    # Fallback to DPM-Solver++ if integrator doesn't have step method
                                    logger.warning(f"Integrator {type(self.integrator).__name__} lacks step() method, using DPM-Solver++")
                                    y_pred = _dpm_solver_predict(x_t)
                            else:
                                # Use default DPM-Solver++
                                y_pred = _dpm_solver_predict(x_t)

                            # Apply optimal transport
                            eps = compute_eps_at(t_curr)
                            # Production release: Use thread-safe flag
                            tm = (self._richardson_extrapolated_transport(x_t, y_pred, eps)
                                  if use_richardson and self._should_use_richardson(score)
                                  else self._standard_transport(x_t, y_pred, eps))
                            x_t = tm(x_t)

                            # Apply corrector step if enabled
                            if self.corrector is not None:
                                def score_fn_wrapper(x_val, t_val):
                                    return self._compute_score_optimized(x_val, t_val)

                                if isinstance(self.corrector, AdaptiveCorrector):
                                    # Adaptive corrector needs previous sample for error estimation
                                    x_prev = y_pred if i > 0 else None
                                    x_t = self.corrector.correct(x_t, t_next, score_fn_wrapper, x_prev, generator)
                                else:
                                    # Regular corrector
                                    if isinstance(self.corrector, TweedieCorrector):
                                        x_t = self.corrector.correct(x_t, t_next, score_fn_wrapper)
                                    else:
                                        x_t = self.corrector.correct(x_t, t_next, score_fn_wrapper, generator)

                            x_t = self._format_tensor(x_t)

                        self.timing_stats['integration'].append(time.time() - integration_start)
                        self.timing_stats['total_step'].append(time.time() - step_start)
                        
                        if verbose and i % 3 == 0 and hasattr(iterator, 'set_postfix_str'):
                            avg_step = np.mean(list(self.timing_stats['total_step']))
                            eta = avg_step * (len(timesteps) - i - 1)
                            iterator.set_postfix_str(f"t={t_next:.3f}, ETA={eta:.1f}s")
                    
                    if output_dtype is not None:
                        x_t = x_t.to(output_dtype)
                    x_t = self._format_tensor(x_t)
                    
                    stats = None
                    if return_stats or use_result_dataclass:
                        # Production release: Surface Sinkhorn backend in stats
                        stats = {
                            'version': __version__,
                            'steps': len(timesteps) - 1,
                            'avg_score_time': np.mean(list(self.timing_stats['score_eval'])) if self.timing_stats['score_eval'] else 0,
                            'avg_integration_time': np.mean(list(self.timing_stats['integration'])) if self.timing_stats['integration'] else 0,
                            'total_time': sum(self.timing_stats['total_step']) if self.timing_stats['total_step'] else 0,
                            'fallback_count': self.fallback_count,
                            'memory_deltas': list(self.memory_stats) if self.config.profile_memory else None,
                            'sinkhorn_backend': self.sinkhorn_kernel.last_backend_used
                        }
                    
                    if use_result_dataclass:
                        return SamplingResult(samples=x_t, stats=stats)
                    elif return_stats:
                        return x_t, stats
                    else:
                        return x_t
                    
        finally:
            pass  # No config mutation to restore
    
    def _richardson_extrapolated_transport(self, x, y, eps):
        """Richardson extrapolation for unbiased OT with performance budget."""
        B = x.size(0)
        if B == 1 and self._patch_transport is not None and not self.config.force_per_pixel_b1:
            return self._patch_transport(x, y, eps)
        
        eps1 = eps
        eps2 = eps / 2
        
        start_time = time.perf_counter()
        
        transport1 = self._standard_transport(x, y, eps1)

        first_solve_time = time.perf_counter() - start_time

        transport2 = self._standard_transport(x, y, eps2)

        total_time = time.perf_counter() - start_time
        # Overhead = (second_solve_time) / (first_solve_time)
        # If second solve costs same as first, overhead = 100% (doubling the work)
        overhead = (total_time - first_solve_time) / max(first_solve_time, 1e-9)
        
        if overhead > self.config.richardson_max_overhead:
            logger.debug(f"Skipping Richardson extrapolation - overhead {overhead:.1%} exceeds budget {self.config.richardson_max_overhead:.1%}")
            self.timing_stats['richardson_overhead'].append(overhead)
            return transport1
        
        self.timing_stats['richardson_overhead'].append(overhead)
        
        def extrapolated_transport(z):
            t1 = transport1(z)
            t2 = transport2(z)
            return 2 * t2 - t1
        
        return extrapolated_transport
    
    def sample(self, *args, **kwargs):
        """Deprecated: Use sample_enhanced() instead."""
        logger.warning("sample() is deprecated and will be removed in a future major release. Use sample_enhanced() instead.")
        return self.sample_enhanced(*args, **kwargs)
    
    def selftest(self, verbose: bool = True) -> Dict[str, Any]:
        """Run comprehensive self-tests to validate correctness.

        Forces deterministic mode for reliable testing.
        """
        results: Dict[str, Any] = {"status": "passed", "tests": {}, "timings": {}}

        old_det = self.config.deterministic
        old_det_cdist = self.config.deterministic_cdist_cpu
        old_rich = self.config.richardson_extrapolation

        try:
            self.config.deterministic = True
            self.config.deterministic_cdist_cpu = True
            self.config.richardson_extrapolation = False

            t_vals = torch.linspace(0, 1, 10, device=self.device)

            # Test 1: lambda(t) is always float32
            test_start = time.time()
            schedules = [
                ("current", self.noise_schedule),
                ("cosine_fp16", CosineSchedule(device=self.device, dtype=torch.float16)),
                ("linear_fp16", LinearSchedule(device=self.device, dtype=torch.float16)),
            ]
            for schedule_name, schedule in schedules:
                lambda_result = schedule.lambda_(0.5)
                test_passed = lambda_result.dtype == torch.float32
                results["tests"][f"{schedule_name}_lambda_fp32"] = bool(test_passed)
                if not test_passed:
                    results["status"] = "failed"
                    logger.error(
                        "CRITICAL: %s lambda returned dtype %s, expected torch.float32",
                        schedule_name,
                        lambda_result.dtype,
                    )
                if verbose:
                    status_text = "PASS" if test_passed else "FAIL"
                    logger.info(
                        "   [%s] %s: lambda(t) dtype = %s (expected torch.float32)",
                        status_text,
                        schedule_name,
                        lambda_result.dtype,
                    )
            results["timings"]["lambda_test"] = time.time() - test_start

            # Test 2: lambda(t) monotonicity
            test_start = time.time()
            lambda_vals = torch.stack([self.noise_schedule.lambda_(float(t)) for t in t_vals]).squeeze(-1)
            is_monotonic = torch.all(lambda_vals[:-1] >= lambda_vals[1:]).item()
            results["tests"]["lambda_monotonic"] = bool(is_monotonic)
            if not is_monotonic:
                results["status"] = "failed"
            if verbose:
                status_text = "PASS" if is_monotonic else "FAIL"
                logger.info(
                    "   [%s] lambda(t) monotonicity: lambda(0)=%.3f >= lambda(1)=%.3f",
                    status_text,
                    lambda_vals[0].item(),
                    lambda_vals[-1].item(),
                )
            results["timings"]["monotonicity_test"] = time.time() - test_start

            # Test 3: alpha^2 + sigma^2 ~= 1
            test_start = time.time()
            alpha, sigma = self.noise_schedule.alpha_sigma(t_vals)
            sum_squares = torch.square(alpha.float()) + torch.square(sigma.float())
            max_error = (sum_squares - 1).abs().max().item()
            test_passed = max_error < 1e-5
            results["tests"]["alpha_sigma_unity"] = bool(test_passed)
            results["tests"]["alpha_sigma_max_error"] = max_error
            if not test_passed:
                results["status"] = "failed"
            if verbose:
                status_text = "PASS" if test_passed else "FAIL"
                logger.info(
                    "   [%s] alpha^2 + sigma^2 ~= 1: max error = %.2e",
                    status_text,
                    max_error,
                )
            results["timings"]["unity_test"] = time.time() - test_start

            # Test 4: Determinism
            if verbose:
                logger.info("   Testing determinism...")
            test_start = time.time()
            shape = (1, 3, 32, 32)

            x1 = self.sample_enhanced(shape, num_steps=2, verbose=False, seed=42)
            x2 = self.sample_enhanced(shape, num_steps=2, verbose=False, seed=42)
            is_deterministic = torch.allclose(x1, x2, atol=0, rtol=0)
            results["tests"]["deterministic_sampling"] = bool(is_deterministic)
            if not is_deterministic:
                results["status"] = "failed"
            if verbose:
                status_text = "PASS" if is_deterministic else "FAIL"
                logger.info(
                    "   [%s] Deterministic sampling (same solver instance)",
                    status_text,
                )
            results["timings"]["determinism_test"] = time.time() - test_start

            # Test 5: Small image support
            if verbose:
                logger.info("   Testing small image support...")
            test_start = time.time()
            small_sizes = [(1, 3, 16, 16), (1, 3, 32, 32), (1, 3, 8, 8)]
            all_work = True
            for sz in small_sizes:
                try:
                    x_small = self.sample_enhanced(sz, num_steps=2, verbose=False, seed=42)
                    if x_small.shape != sz:
                        logger.debug("Small image %s returned shape %s", sz, tuple(x_small.shape))
                        all_work = False
                        break
                except Exception as small_exc:  # pragma: no cover - diagnostic logging
                    logger.debug("Small image %s failed: %s", sz, small_exc)
                    all_work = False
                    break

            results["tests"]["small_image_support"] = all_work
            if not all_work:
                results["status"] = "failed"
            if verbose:
                status_text = "PASS" if all_work else "FAIL"
                logger.info(
                    "   [%s] Small image support (8x8, 16x16, 32x32)",
                    status_text,
                )
            results["timings"]["small_image_test"] = time.time() - test_start

        except Exception as exc:
            results["status"] = "error"
            results["error"] = str(exc)
            if verbose:
                logger.error("   [FAIL] Self-test terminated with exception: %s", exc)
        finally:
            self.config.deterministic = old_det
            self.config.deterministic_cdist_cpu = old_det_cdist
            self.config.richardson_extrapolation = old_rich

        if verbose:
            summary = f"\n[SELFTEST] Result: {results['status'].upper()}"
            logger.info(summary)
            if results["status"] == "passed":
                total_time = sum(results["timings"].values()) if results["timings"] else 0.0
                logger.info("SPOT %s is production-ready!", __version__)
                logger.info("   Total test time: %.3fs", total_time)

        return results
    
    def cleanup(self):
        """Clean up resources and restore global settings."""
        self._restore_tf32_settings()
        logger.debug("SPOT solver cleanup complete")
