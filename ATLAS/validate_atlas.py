#!/usr/bin/env python3
"""
Lightweight readiness checks for the ATLAS toolkit.

The script runs a curated collection of smoke tests that cover:
* Core imports
* Configuration objects
* Model forward passes
* Solver and sampler steps
* Easy API validation
* Utility helpers

It is intentionally CPU-only and keeps tensor sizes tiny so it can be run on
development laptops as part of pre-flight verification.
"""

from __future__ import annotations

from pathlib import Path
import sys
import traceback
from typing import Callable, Iterable, List, Sequence, Tuple

import torch


PASSED: List[str] = []
FAILED: List[Tuple[str, Exception]] = []
DEVICE = torch.device("cpu")


def log_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80)


def run_test(name: str, func: Callable[[], None]) -> None:
    print(f" - {name} ... ", end="", flush=True)
    try:
        func()
    except Exception as exc:  # pragma: no cover - diagnostic path
        print("FAIL")
        FAILED.append((name, exc))
        traceback.print_exc()
    else:
        print("OK")
        PASSED.append(name)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def tiny_model_config():
    from atlas.config import HighResModelConfig

    return HighResModelConfig(
        in_channels=2,
        out_channels=2,
        base_channels=32,
        channel_mult=(1,),
        num_res_blocks=1,
        attention_levels=(),
        cross_attention_levels=(),
        conditional=False,
        use_clip_conditioning=False,
        context_dim=0,
    )


def tiny_kernel_config():
    from atlas.config import KernelConfig

    return KernelConfig(
        kernel_type="gaussian",
        epsilon=0.05,
        solver_type="direct",
        rff_features=64,
        n_landmarks=16,
        max_kernel_cache_size=4,
        multi_scale=False,
        scale_factors=[1.0],
    )


def tiny_sampler_config():
    from atlas.config import SamplerConfig

    return SamplerConfig(
        sb_iterations=1,
        hierarchical_sampling=False,
        use_linear_solver=False,
        memory_efficient=False,
        use_mixed_precision=False,
        verbose_logging=False,
        auto_tuning=False,
    )


def build_tiny_model():
    from atlas.models import HighResLatentScoreModel

    config = tiny_model_config()
    model = HighResLatentScoreModel(config).to(DEVICE)
    model.eval()
    return model, config


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def import_tests():
    def test_main_package():
        import atlas  # noqa: F401

        assert hasattr(atlas, "__version__")

    def test_config_package():
        from atlas.config import (  # noqa: F401
            ConditioningConfig,
            DatasetConfig,
            HighResModelConfig,
            InferenceConfig,
            KernelConfig,
            LoRAConfig,
            SamplerConfig,
            TrainingConfig,
            presets,
        )

        assert presets is not None

    def test_model_package():
        from atlas.models import HighResLatentScoreModel, build_highres_score_model

        cfg = tiny_model_config()
        model = HighResLatentScoreModel(cfg)
        auto = build_highres_score_model(cfg)
        assert isinstance(model, type(auto))

    def test_solver_package():
        from atlas.solvers import AdvancedHierarchicalDiffusionSampler, SchroedingerBridgeSolver

        assert AdvancedHierarchicalDiffusionSampler is not None
        assert SchroedingerBridgeSolver is not None

    def test_kernel_package():
        from atlas.kernels import DirectKernelOperator, FFTKernelOperator, NystromKernelOperator, RFFKernelOperator

        assert len({DirectKernelOperator, FFTKernelOperator, NystromKernelOperator, RFFKernelOperator}) == 4

    def test_utils_package():
        from atlas.utils import (
            CUDAGraphModelWrapper,
            NoisePredictionAdapter,
            TiledModelWrapper,
            build_dataset,
            create_dataloader,
            override_dataset_root,
            set_seed,
        )

        for symbol in (
            CUDAGraphModelWrapper,
            NoisePredictionAdapter,
            TiledModelWrapper,
            build_dataset,
            create_dataloader,
            override_dataset_root,
            set_seed,
        ):
            assert callable(symbol) or isinstance(symbol, type)

    return [
        ("Main package import", test_main_package),
        ("Config imports", test_config_package),
        ("Model imports", test_model_package),
        ("Solver imports", test_solver_package),
        ("Kernel imports", test_kernel_package),
        ("Utils imports", test_utils_package),
    ]


def configuration_tests():
    def test_config_creation():
        from atlas.config import DatasetConfig, InferenceConfig, KernelConfig, SamplerConfig, TrainingConfig

        dataset = DatasetConfig(
            name="lsun",
            root="./data/lsun",
            resolution=64,
            channels=3,
            batch_size=4,
        )
        train = TrainingConfig(batch_size=8, micro_batch_size=4, learning_rate=1e-4)
        infer = InferenceConfig(batch_size=2, sampler_steps=4)
        kernel = KernelConfig()
        sampler = SamplerConfig()

        assert dataset.batch_size == 4
        assert train.batch_size // (train.micro_batch_size or 1) >= 2
        assert infer.sampler_steps > 0
        assert kernel.kernel_type in {"gaussian", "laplacian", "cauchy"}
        assert sampler.sb_iterations > 0

    def test_set_seed():
        from atlas.utils import set_seed

        set_seed(42)
        a = torch.rand(4)
        set_seed(42)
        b = torch.rand(4)
        assert torch.allclose(a, b)

    return [
        ("Config dataclasses", test_config_creation),
        ("Seed helper", test_set_seed),
    ]


def model_tests():
    def test_forward_pass():
        model, config = build_tiny_model()
        with torch.no_grad():
            x = torch.randn(2, config.in_channels, 8, 8, device=DEVICE)
            t = torch.linspace(0.0, 1.0, 2, device=DEVICE)
            out = model(x, t)
        assert out.shape == x.shape

    return [("HighResLatentScoreModel forward", test_forward_pass)]


def solver_and_sampler_tests():
    def test_solver_sampling():
        from atlas.schedules import karras_noise_schedule
        from atlas.solvers import SchroedingerBridgeSolver

        model, config = build_tiny_model()
        kernel_cfg = tiny_kernel_config()
        sampler_cfg = tiny_sampler_config()

        solver = SchroedingerBridgeSolver(
            score_model=model,
            noise_schedule=karras_noise_schedule,
            device=DEVICE,
            kernel_config=kernel_cfg,
            sampler_config=sampler_cfg,
        )

        timesteps = [1.0, 0.5, 0.0]
        with torch.no_grad():
            samples = solver.sample((1, config.in_channels, 8, 8), timesteps, verbose=False)
        assert samples.shape == (1, config.in_channels, 8, 8)

    def test_hierarchical_sampler():
        from atlas.schedules import karras_noise_schedule
        from atlas.solvers import AdvancedHierarchicalDiffusionSampler

        model, config = build_tiny_model()
        sampler = AdvancedHierarchicalDiffusionSampler(
            score_model=model,
            noise_schedule=karras_noise_schedule,
            device=DEVICE,
            kernel_config=tiny_kernel_config(),
            sampler_config=tiny_sampler_config(),
        )

        with torch.no_grad():
            result = sampler.sample(
                shape=(1, config.in_channels, 8, 8),
                timesteps=[1.0, 0.5, 0.0],
                verbose=False,
            )
        assert result.shape == (1, config.in_channels, 8, 8)

    return [
        ("SchroedingerBridgeSolver sample", test_solver_sampling),
        ("AdvancedHierarchicalDiffusionSampler sample", test_hierarchical_sampler),
    ]


def easy_api_tests():
    def test_validate_configs():
        from atlas.config import ConditioningConfig
        from atlas.easy_api import GPUProfile, validate_configs

        profile = GPUProfile(
            name="test",
            memory_mb=2048,
            batch_size=2,
            resolution=512,
            use_mixed_precision=False,
            kernel_solver="rff",
            kernel_cache_size=4,
            enable_clip=False,
            gradient_checkpointing=False,
            description="test profile",
        )

        warnings_list = validate_configs(
            model_config=tiny_model_config(),
            kernel_config=tiny_kernel_config(),
            sampler_config=tiny_sampler_config(),
            conditioning_config=ConditioningConfig(use_clip=False),
            gpu_profile=profile,
        )
        assert isinstance(warnings_list, list)

    return [("easy_api.validate_configs", test_validate_configs)]


def utility_tests():
    def test_noise_schedule():
        from atlas.schedules import karras_noise_schedule

        values = torch.linspace(0, 1, 5)
        alphas = karras_noise_schedule(values)
        assert torch.all((alphas > 0) & (alphas <= 1))

    def test_memory_helpers():
        from atlas.utils import get_peak_memory_mb, reset_peak_memory

        reset_peak_memory()
        before = get_peak_memory_mb()
        torch.randn(8, 8)  # allocate briefly
        after = get_peak_memory_mb()
        assert after >= before

    return [
        ("karras_noise_schedule output range", test_noise_schedule),
        ("Memory helper utilities", test_memory_helpers),
    ]


SECTIONS: Sequence[Tuple[str, Iterable[Tuple[str, Callable[[], None]]]]] = [
    ("Import verification", import_tests()),
    ("Configuration basics", configuration_tests()),
    ("Model checks", model_tests()),
    ("Solvers & samplers", solver_and_sampler_tests()),
    ("Easy API", easy_api_tests()),
    ("Utility helpers", utility_tests()),
]


def main() -> int:
    log_header("ATLAS COMPREHENSIVE VALIDATION")
    project_root = Path(__file__).resolve().parent
    print(f"Project root: {project_root}")
    print(f"Device used : {DEVICE}")

    for title, tests in SECTIONS:
        log_header(title)
        for name, func in tests:
            run_test(name, func)

    log_header("SUMMARY")
    print(f"Passed: {len(PASSED)}")
    print(f"Failed: {len(FAILED)}")

    if FAILED:
        print("\nFailing tests:")
        for name, exc in FAILED:
            print(f" * {name}: {exc}")
        return 1

    print("\nATLAS is ready for training and sampling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
