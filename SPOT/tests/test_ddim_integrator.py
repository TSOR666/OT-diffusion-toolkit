import torch

from SPOT.integrators import DDIMIntegrator


class DummySchedule:
    def alpha_sigma(self, t):
        t_tensor = torch.as_tensor(t, dtype=torch.float32)
        alpha_low = torch.full_like(t_tensor, 0.6)
        sigma_low = torch.full_like(t_tensor, 0.8)
        alpha_high = torch.full_like(t_tensor, 0.8)
        sigma_high = torch.full_like(t_tensor, 0.6)
        alpha = torch.where(t_tensor > 0.5, alpha_low, alpha_high)
        sigma = torch.where(t_tensor > 0.5, sigma_low, sigma_high)
        return alpha, sigma


def _ones_score(x, t):
    return torch.ones_like(x)


def test_ddim_deterministic_uses_score_to_noise_conversion():
    schedule = DummySchedule()
    integrator = DDIMIntegrator(schedule, eta=0.0)

    x = torch.ones(1, 1, 2, 2)
    x_next = integrator.step(x, t_curr=0.75, t_next=0.25, score_fn=_ones_score)

    alpha_curr = torch.tensor(0.6)
    sigma_curr = torch.tensor(0.8)
    alpha_next = torch.tensor(0.8)
    sigma_next = torch.tensor(0.6)

    pred_x0 = (x + sigma_curr**2) / alpha_curr
    epsilon = -sigma_curr
    expected = alpha_next * pred_x0 + sigma_next * epsilon

    torch.testing.assert_close(x_next, expected, rtol=1e-6, atol=1e-6)
