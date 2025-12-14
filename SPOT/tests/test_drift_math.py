import torch

from SPOT.integrators import EulerIntegrator, HeunIntegrator


class LargeSigmaSchedule:
    """Schedule with constant beta and large sigma to expose incorrect scaling."""

    def alpha_sigma(self, t):
        t_tensor = torch.as_tensor(t, dtype=torch.float32)
        alpha = torch.ones_like(t_tensor)
        sigma = torch.full_like(t_tensor, 10.0)  # Large sigma would dominate if incorrectly used
        return alpha, sigma

    def lambda_(self, t):
        t_tensor = torch.as_tensor(t, dtype=torch.float32)
        return torch.zeros_like(t_tensor)

    def beta(self, t):
        t_tensor = torch.as_tensor(t, dtype=torch.float32)
        return torch.full_like(t_tensor, 2.0)  # Constant beta


def _ones_score(x, t):
    return torch.ones_like(x)


def test_heun_integrator_uses_pf_ode_drift():
    schedule = LargeSigmaSchedule()
    integrator = HeunIntegrator(schedule)
    x0 = torch.ones(1, 1, 2, 2)

    # dt < 0 to mirror reverse-time integration
    x_next = integrator.step(x0, t_curr=0.5, t_next=0.4, score_fn=_ones_score)

    # Heun's method (predictor-corrector, 2nd order Runge-Kutta):
    # Probability-flow drift: f(x,t) = -0.5*beta*x - beta*score with beta=2, score=1
    #
    # Step 1 (predictor): Euler step
    #   drift_curr = -0.5*2*1 - 2*1 = -3
    #   x_pred = 1 + (-3)*(-0.1) = 1.3
    #
    # Step 2 (corrector): Evaluate drift at predicted point
    #   drift_next = -0.5*2*1.3 - 2*1 = -1.3 - 2 = -3.3
    #
    # Step 3: Trapezoidal rule (average of drifts)
    #   x_next = 1 + 0.5*(-3 + -3.3)*(-0.1) = 1 + 0.5*(-6.3)*(-0.1) = 1.315
    expected = torch.full_like(x0, 1.315)
    torch.testing.assert_close(x_next, expected, rtol=1e-5, atol=1e-5)


def test_euler_integrator_uses_pf_ode_drift():
    schedule = LargeSigmaSchedule()
    integrator = EulerIntegrator(schedule)
    x0 = torch.ones(1, 1, 2, 2)

    x_next = integrator.step(x0, t_curr=0.5, t_next=0.4, score_fn=_ones_score)
    expected = torch.full_like(x0, 1.3)
    torch.testing.assert_close(x_next, expected, rtol=1e-6, atol=1e-6)
