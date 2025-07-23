from torch import nn, linspace, cumprod, randn_like, sqrt


class DDPM_Scheduler:
    def __init__(self, num_time_steps: int = 1000):
        super().__init__()
        self.beta = linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = cumprod(alpha, dim=0).requires_grad_(False)

    def noise_frame(self, device, frame, steps):
        e = randn_like(frame, requires_grad=False).to(device)
        a = self.alpha[steps].view(len(steps), 1, 1, 1).to(device)
        return (sqrt(a) * frame) + (sqrt(1 - a) * e)
