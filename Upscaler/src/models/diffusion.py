import torch
from torch import save, nn, logical_not, randn_like, ones_like, cat
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from .unet import WithSinusoidalEmbedding
from monai.losses import SSIMLoss, MaskedLoss
from tqdm import tqdm
from .util import display_images


class Model:
    def __init__(self, lr, device=None, warmup_steps=5000):
        # Scheduling
        self.total_timesteps = 1000
        s = 0.008
        steps = self.total_timesteps
        t = torch.linspace(0, steps, steps + 1, device=device)
        f = torch.cos((t / steps + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bar_cosine = f / f[0]
        self.beta = torch.clamp(
            1 - alpha_bar_cosine[1:] / alpha_bar_cosine[:-1], max=0.999
        ).to(device)
        self.alpha = (1.0 - self.beta).to(device)
        self.alpha_bar = alpha_bar_cosine[1:].to(device)

        self.model = WithSinusoidalEmbedding(
            input_channels=3, time_steps=self.total_timesteps
        ).to(device)

        self.ema_model = WithSinusoidalEmbedding(
            input_channels=3, time_steps=self.total_timesteps
        ).to(device)

        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_decay = 0.999

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        warmup = lambda step: min(1.0, step / warmup_steps)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=warmup)
        self.loss = nn.HuberLoss(reduction="none")
        self.ssim = MaskedLoss(
            loss=SSIMLoss(spatial_dims=2, data_range=2.0, reduction="none")
        )

    def load(self, file, device=None):
        try:
            print("Trying to load", file)
            if file is not None:
                checkpoint = torch.load(file, map_location=device)
                print("Loaded checkpoint")
                self.model.load_state_dict(checkpoint["model"])
                print("Loaded model")
                self.ema_model.load_state_dict(checkpoint["ema_model"])
                print("Loaded EMA model")
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                print("Loaded optimizer")
        except Exception as e:
            print("Could not load checkpoint", e)

    def checkpoint(self, checkpoint_path):
        checkpoint = {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        print("Saving checkpoint to", checkpoint_path)

        save(checkpoint, checkpoint_path)

        print("Saved checkpoint to", checkpoint_path)

    def noise_input_at_step(self, i, t):
        noise = torch.randn_like(i)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        return (torch.sqrt(alpha_bar_t) * i) + (
            torch.sqrt(1.0 - alpha_bar_t) * noise
        ), noise

    @torch.no_grad()
    def infer(self, data, mask, num_inference_steps=200, device=None):
        self.ema_model.eval()

        data_masked = data * mask
        data_heights = (data_masked[:, 0:1, :, :] * 2) - 1

        batch_size = data.shape[0]
        x_t = randn_like(data_heights, device=device)

        alpha_bar = self.alpha_bar

        step_indices = torch.linspace(
            self.total_timesteps - 1, 0, num_inference_steps, device=device
        ).long()

        for i in tqdm(range(len(step_indices) - 1), position=2):
            t = step_indices[i].item()
            t_prev = step_indices[i + 1].item()

            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            noised_input, _ = self.noise_input_at_step(data_heights, t_tensor)

            x_t = (mask * noised_input) + (logical_not(mask) * x_t)

            model_input = cat([x_t, data_heights, mask.float()], dim=1)

            inferred_noise = self.ema_model.forward(model_input, ones_like(mask), t_tensor)

            # DDIM deterministic update
            alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)
            alpha_bar_t_prev = alpha_bar[t_prev].view(-1, 1, 1, 1)

            # Predict the clean image from the estimated noise
            x_0_pred = (
                x_t - torch.sqrt(1.0 - alpha_bar_t) * inferred_noise
            ) / torch.sqrt(alpha_bar_t)
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

            # Jump directly to the previous timestep (deterministic, no noise added)
            x_t = (
                torch.sqrt(alpha_bar_t_prev) * x_0_pred
                + torch.sqrt(1.0 - alpha_bar_t_prev) * inferred_noise
            )

        denoised_output = x_t
        denoised_output = (denoised_output + 1) / 2

        # images.append(denoised_output.detach().to("cpu"))
        # display_images(images, to_file="./infer")

        self.ema_model.train()
        return denoised_output

    def ema_update(self):
        with torch.no_grad():
            for ema_p, model_p in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_p.data.mul_(self.ema_decay).add_(
                    model_p.data, alpha=(1 - self.ema_decay)
                )

    def train_step(self, data, mask, expected, expected_good_mask):
        active_learning_region = expected_good_mask * logical_not(mask)
        if active_learning_region.sum() == 0:
            print("Zero active region — skipping step")
            return torch.zeros(1, device=data.device, requires_grad=False)

        self.optimizer.zero_grad(set_to_none=True)

        data = (data * 2) - 1
        expected = (expected * 2) - 1

        batch_size = data.shape[0]
        timestep = torch.randint(
            0, self.total_timesteps, (batch_size,), device=data.device
        ).long()

        alpha_bar_t = self.alpha_bar[timestep].view(-1, 1, 1, 1)
        noised_target, noise_added = self.noise_input_at_step(expected, timestep)

        model_input = cat([noised_target, data, mask.float()], dim=1)

        inferred_noise = self.model.forward(model_input, expected_good_mask, timestep)

        noise_loss = self.loss(inferred_noise, noise_added) * active_learning_region
        noise_loss = noise_loss.sum() / (active_learning_region.sum() + 1e-8)

        reconstructed_image = (
            noised_target - torch.sqrt(1.0 - alpha_bar_t) * inferred_noise
        ) / torch.sqrt(alpha_bar_t)

        rec_weight = alpha_bar_t.view(-1, 1, 1, 1)
        reconstructed_image_loss = (
            self.loss(reconstructed_image, expected)
            * active_learning_region
            * rec_weight
        )
        reconstructed_image_loss = reconstructed_image_loss.sum() / (
            active_learning_region.sum() + 1e-8
        )

        reconstructed_clamped = torch.clamp(reconstructed_image, -1.0, 1.0)
        ssim_loss = self.ssim(
            reconstructed_clamped, expected, mask=active_learning_region.float()
        ).squeeze()
        ssim_weight = alpha_bar_t.view(-1)
        ssim_loss = (ssim_loss * ssim_weight).mean()

        loss = noise_loss
        loss += reconstructed_image_loss * 0.1
        loss += ssim_loss * 0.05

        if not torch.isfinite(loss).item():
            err = []
            for name, val in [("noise", noise_loss), ("recon", reconstructed_image_loss), ("ssim", ssim_loss)]:
                if not torch.isfinite(val).all():
                    err.append(name)
            print(f"NaN/Inf loss — skipping ({', '.join(err)} component)")
            return torch.zeros(1, device=loss.device, requires_grad=False)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.ema_update()

        return loss
