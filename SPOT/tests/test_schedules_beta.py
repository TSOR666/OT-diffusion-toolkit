import torch

from SPOT.constants import EPSILON_CLAMP
from SPOT.schedules import CosineSchedule, LinearSchedule


def _finite_difference_beta(schedule, t_scalar: float, delta: float = 1e-4) -> torch.Tensor:
    """Numerically approximate -d(lambda)/dt via central difference on lambda(t)."""
    t_plus = torch.tensor([min(1.0, t_scalar + delta)], dtype=torch.float32)
    t_minus = torch.tensor([max(0.0, t_scalar - delta)], dtype=torch.float32)
    lambda_plus = schedule.lambda_(t_plus)
    lambda_minus = schedule.lambda_(t_minus)
    return -((lambda_plus - lambda_minus) / (t_plus - t_minus))


def _linear_lambda_derivative(schedule, t_scalar: float) -> torch.Tensor:
    t = torch.tensor([t_scalar], dtype=torch.float64, requires_grad=True)
    beta0, beta1 = schedule.beta_start, schedule.beta_end
    integral = beta0 * t + 0.5 * (beta1 - beta0) * t * t
    alpha_bar = torch.exp(-integral)
    sigma_sq = (1.0 - alpha_bar).clamp_min(EPSILON_CLAMP)
    lambda_val = torch.log(alpha_bar / sigma_sq)
    lambda_val.backward()
    return -t.grad.float()


def test_linear_schedule_beta_matches_lambda_derivative():
    schedule = LinearSchedule()
    t = 0.5
    beta_analytic = schedule.beta(torch.tensor([t], dtype=torch.float32))
    beta_autograd = _linear_lambda_derivative(schedule, t)
    torch.testing.assert_close(beta_analytic, beta_autograd, rtol=1e-4, atol=1e-4)


def test_cosine_schedule_beta_matches_lambda_derivative():
    schedule = CosineSchedule()
    t = 0.3
    beta_analytic = schedule.beta(torch.tensor([t], dtype=torch.float32))
    beta_fd = _finite_difference_beta(schedule, t)
    torch.testing.assert_close(beta_analytic, beta_fd, rtol=1e-3, atol=1e-3)


def test_linear_schedule_beta_dtype_invariant():
    t = 0.7
    beta_fp32 = LinearSchedule(dtype=torch.float32).beta(torch.tensor([t], dtype=torch.float32))
    beta_fp16 = LinearSchedule(dtype=torch.float16).beta(torch.tensor([t], dtype=torch.float32))
    torch.testing.assert_close(beta_fp16, beta_fp32, rtol=1e-5, atol=1e-5)
