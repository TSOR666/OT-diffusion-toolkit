"""Hilbert Sinkhorn divergence implementation."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .fft_ot import FFTOptimalTransport
from .kernel import KernelDerivativeRFF


class HilbertSinkhornDivergence:
    """Implements the Hilbert Sinkhorn Divergence for optimal transport."""

    def __init__(
        self,
        epsilon: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-5,
        device: torch.device | None = None,
        adaptive_epsilon: bool = True,
        kernel_type: str = "gaussian",
        sigma: float = 1.0,
        debiased: bool = True,
        use_rff: bool = True,
        rff_features: int = 1024,
        accelerated: bool = False,
    ) -> None:
        # Input validation
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if rff_features < 1:
            raise ValueError(f"rff_features must be at least 1, got {rff_features}")
        if kernel_type not in ["gaussian", "laplacian"]:
            raise ValueError(
                f"kernel_type must be one of ['gaussian', 'laplacian'], "
                f"got '{kernel_type}'. Note: 'cauchy' kernel is not supported by FFT-OT backend."
            )

        # When use_rff=True, only Gaussian kernels are supported by the RFF implementation
        if use_rff and kernel_type != "gaussian":
            raise NotImplementedError(
                f"Random Fourier Features (use_rff=True) currently only supports 'gaussian' kernel. "
                f"For kernel type '{kernel_type}', either set use_rff=False to use exact kernel computation, "
                f"or use kernel_type='gaussian'."
            )

        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adaptive_epsilon = adaptive_epsilon
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.debiased = debiased
        self.use_rff = use_rff
        self.rff_features = rff_features
        self.accelerated = accelerated

        self.rff: KernelDerivativeRFF | None = None
        self.fft_ot = FFTOptimalTransport(
            epsilon=epsilon,
            max_iter=max_iter,
            tol=tol,
            kernel_type=kernel_type,
            device=self.device,
            multiscale=True,
        )

    def _initialize_rff(self, dim: int) -> None:
        if self.rff is None or self.rff.input_dim != dim:
            self.rff = KernelDerivativeRFF(
                input_dim=dim,
                feature_dim=self.rff_features,
                sigma=self.sigma,
                kernel_type=self.kernel_type,
                device=self.device,
                orthogonal=True,
            )

    def _compute_cost_matrix(
        self, x: torch.Tensor, y: torch.Tensor, use_rff: Optional[bool] = None
    ) -> torch.Tensor:
        use_rff = self.use_rff if use_rff is None else use_rff

        if use_rff:
            dim = x.size(1) if x.dim() > 1 else 1
            self._initialize_rff(dim)
            if self.rff is None:
                raise RuntimeError("Failed to initialize RFF kernel")
            x_features = self.rff.compute_features(x)
            y_features = self.rff.compute_features(y)
            x_norm = (x_features**2).sum(1, keepdim=True)
            y_norm = (y_features**2).sum(1, keepdim=True)
            xy = x_features @ y_features.T
            cost_matrix = x_norm + y_norm.T - 2 * xy
            cost_matrix = torch.clamp(cost_matrix, min=0.0)
            # Scale by sigma^2 to approximate Euclidean cost in the original space
            cost_matrix = cost_matrix * (self.sigma ** 2)
        else:
            cost_matrix = torch.cdist(x, y, p=2).pow(2)
        return cost_matrix

    def _sinkhorn_algorithm(
        self,
        cost_matrix: torch.Tensor,
        weights_x: Optional[torch.Tensor] = None,
        weights_y: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = eps or self.epsilon
        nx, ny = cost_matrix.shape

        # Handle empty tensor edge cases
        if nx == 0 or ny == 0:
            u = torch.zeros(nx, device=cost_matrix.device)
            v = torch.zeros(ny, device=cost_matrix.device)
            transport = torch.zeros((nx, ny), device=cost_matrix.device)
            return u, v, transport

        if weights_x is None:
            weights_x = torch.ones(nx, device=cost_matrix.device) / nx
        else:
            sum_x = weights_x.sum()
            if sum_x > 0:
                weights_x = weights_x / sum_x
            else:
                weights_x = torch.ones(nx, device=cost_matrix.device) / nx
        if weights_y is None:
            weights_y = torch.ones(ny, device=cost_matrix.device) / ny
        else:
            sum_y = weights_y.sum()
            if sum_y > 0:
                weights_y = weights_y / sum_y
            else:
                weights_y = torch.ones(ny, device=cost_matrix.device) / ny

        u = torch.zeros(nx, device=cost_matrix.device)
        v = torch.zeros(ny, device=cost_matrix.device)

        kernel = torch.exp(-cost_matrix / eps)
        log_kernel = -cost_matrix / eps
        log_weights_x = torch.log(weights_x)
        log_weights_y = torch.log(weights_y)

        for _ in range(self.max_iter):
            u_prev = u.clone()
            v = log_weights_y - torch.logsumexp(u[:, None] + log_kernel, dim=0)
            u = log_weights_x - torch.logsumexp(v[None, :] + log_kernel, dim=1)
            if torch.max(torch.abs(u - u_prev)) < self.tol:
                break

        plan = torch.exp(u[:, None] + v[None, :]) * kernel
        cost = torch.sum(plan * cost_matrix)
        entropy = -torch.sum(plan * torch.log(plan + 1e-15))
        total_cost = cost - eps * entropy
        return total_cost, u, v

    def _accelerated_sinkhorn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weights_x: Optional[torch.Tensor] = None,
        weights_y: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
    ) -> torch.Tensor:
        is_grid_x, grid_shape_x = self.fft_ot._is_grid_structured(x)
        is_grid_y, grid_shape_y = self.fft_ot._is_grid_structured(y)

        if self.accelerated and is_grid_x and is_grid_y and grid_shape_x == grid_shape_y:
            s_xy, _, _ = self.fft_ot.optimal_transport(x, y, weights_x, weights_y)
            if self.debiased:
                s_xx, _, _ = self.fft_ot.optimal_transport(x, x, weights_x, weights_x)
                s_yy, _, _ = self.fft_ot.optimal_transport(y, y, weights_y, weights_y)
                return s_xy - 0.5 * (s_xx + s_yy)
            return s_xy

        eps = eps or self.epsilon
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        if y.dim() > 2:
            y = y.reshape(y.size(0), -1)

        c_xy = self._compute_cost_matrix(x, y)
        s_xy, _, _ = self._sinkhorn_algorithm(c_xy, weights_x, weights_y, eps)

        if not self.debiased:
            return s_xy

        c_xx = self._compute_cost_matrix(x, x)
        c_yy = self._compute_cost_matrix(y, y)
        s_xx, _, _ = self._sinkhorn_algorithm(c_xx, weights_x, weights_x, eps)
        s_yy, _, _ = self._sinkhorn_algorithm(c_yy, weights_y, weights_y, eps)
        return s_xy - 0.5 * (s_xx + s_yy)

    def compute_divergence(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weights_x: Optional[torch.Tensor] = None,
        weights_y: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
    ) -> torch.Tensor:
        if self.adaptive_epsilon and eps is None:
            if x.dim() > 2:
                x_flat = x.reshape(x.size(0), -1)
                y_flat = y.reshape(y.size(0), -1)
            else:
                x_flat = x
                y_flat = y

            x_sample = x_flat[: min(100, x_flat.size(0))]
            y_sample = y_flat[: min(100, y_flat.size(0))]
            avg_dist = torch.cdist(x_sample, y_sample, p=2).mean()

            eps = float(self.epsilon * (1 + 0.0 * avg_dist))

        return self._accelerated_sinkhorn(x, y, weights_x, weights_y, eps)
