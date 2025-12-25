import torch

from fastsb_ot.config import FastSBOTConfig
from fastsb_ot.solver import FastSBOTSolver, make_schedule


class DummyScore(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Simple, stable score-like output with correct shape.
        return -x


def main() -> None:
    device = torch.device("cpu")
    config = FastSBOTConfig(
        use_mixed_precision=False,
        use_bfloat16=False,
        use_fft_transport=False,
        use_momentum_transport=False,
        use_hierarchical_bridge=False,
        use_fisher_geometry=False,
        corrector_steps=0,
        warmup=False,
    )
    schedule = make_schedule("cosine", num_timesteps=32)
    model = DummyScore()
    solver = FastSBOTSolver(model, schedule, config, device)

    shape = (2, 3, 8, 8)
    timesteps = [1.0, 0.5, 0.0]

    samples = solver.sample(shape, timesteps, verbose=False)
    if samples.shape != shape:
        raise AssertionError(f"Unexpected shape: {samples.shape} != {shape}")
    if not torch.isfinite(samples).all().item():
        raise AssertionError("Non-finite values in samples")

    x = torch.randn(shape, device=device, requires_grad=True)
    score = x.clone()
    dt = torch.tensor(0.1, device=device)
    drift = solver.compute_controlled_drift(x, score, alpha_bar_t=0.5, dt=dt)
    out = solver.compute_drift_and_transport_inline(x, drift, alpha_bar=0.5)
    loss = out.mean()
    loss.backward()
    if x.grad is None or not torch.isfinite(x.grad).all().item():
        raise AssertionError("Non-finite gradients")

    print("VERIFICATION PASSED")


if __name__ == "__main__":
    main()
