import time
from typing import Iterable, Tuple

import torch

from atlas.kernels import (
    DirectKernelOperator,
    FFTKernelOperator,
    NystromKernelOperator,
    RFFKernelOperator,
)


def benchmark_kernel_operators(
    shapes: Iterable[Tuple[int, int]], device: torch.device = torch.device("cpu")
) -> None:
    for batch, dim in shapes:
        x = torch.randn(batch, dim, device=device)
        v = torch.randn(batch, device=device)

        ops = [
            ("direct", DirectKernelOperator(kernel_type="gaussian", epsilon=0.1, device=device)),
            ("rff", RFFKernelOperator(input_dim=dim, feature_dim=64, epsilon=0.1, device=device)),
            (
                "nystrom",
                NystromKernelOperator(
                    landmarks=torch.randn(min(16, batch), dim, device=device),
                    epsilon=0.1,
                    device=device,
                ),
            ),
        ]
        if dim in (4, 9, 16):
            ops.append(("fft", FFTKernelOperator([int(dim ** 0.5), int(dim ** 0.5)], epsilon=0.1, device=device)))

        for name, op in ops:
            start = time.time()
            _ = op.apply(x, v)
            duration = (time.time() - start) * 1000
            print(f"{name:8s} | batch={batch:3d}, dim={dim:3d} -> {duration:6.2f} ms")


if __name__ == "__main__":
    benchmark_kernel_operators([(32, 4), (64, 8), (128, 16)])
