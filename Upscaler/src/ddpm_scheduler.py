from torch import linspace, cumprod, randn_like, sqrt


class DDPM_Scheduler:
    def __init__(self, num_time_steps=None, device=None):
        super().__init__()
        self.beta = linspace(
            1e-4, 0.02, num_time_steps, requires_grad=False, device=device
        )
        alpha = 1 - self.beta
        self.alpha = cumprod(alpha, dim=0).requires_grad_(False)
        self.device = device

    def denoise_from(self, device, frame, predicted_noise, step):
        t = [step]
        beta = self.beta[t]
        alpha = self.alpha[t]
        weight_frame_by = 1 / (sqrt(1 - beta))
        weight_noise_by = beta / ((sqrt(1 - alpha)) * (sqrt(1 - beta)))
        next_frame = (weight_frame_by * frame) - (weight_noise_by * predicted_noise)
        return next_frame

    def noise_frame(self, frame, steps):
        e = randn_like(frame, requires_grad=False, device=self.device)
        a = self.alpha[steps].view(len(steps), 1, 1, 1)
        weight_frame_by = sqrt(a)
        weight_noise_by = sqrt(1 - a)
        noised_frame = (weight_frame_by * frame) + (weight_noise_by * e)
        return noised_frame, e
