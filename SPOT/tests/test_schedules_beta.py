import torch

from SPOT.schedules import CosineSchedule, LinearSchedule


def _finite_difference_beta(schedule, t_scalar: float, delta: float = 1e-4) -> torch.Tensor:
    """Numerically approximate -d(lambda)/dt via central difference on lambda(t)."""
    t_plus = torch.tensor([min(1.0, t_scalar + delta)], dtype=torch.float32)
    t_minus = torch.tensor([max(0.0, t_scalar - delta)], dtype=torch.float32)
    lambda_plus = schedule.lambda_(t_plus)
    lambda_minus = schedule.lambda_(t_minus)
    return -((lambda_plus - lambda_minus) / (t_plus - t_minus))


def test_linear_schedule_beta_matches_lambda_derivative():
    schedule = LinearSchedule()
    t = 0.5
    beta_analytic = schedule.beta(torch.tensor([t], dtype=torch.float32))
    beta_fd = _finite_difference_beta(schedule, t)
    torch.testing.assert_close(beta_analytic, beta_fd, rtol=1e-3, atol=1e-3)


def test_cosine_schedule_beta_matches_lambda_derivative():
    schedule = CosineSchedule()
    t = 0.3
    beta_analytic = schedule.beta(torch.tensor([t], dtype=torch.float32))
    beta_fd = _finite_difference_beta(schedule, t)
    torch.testing.assert_close(beta_analytic, beta_fd, rtol=1e-3, atol=1e-3)
