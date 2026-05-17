import torch
from torch import save, nn, logical_not, randn_like, ones_like, cat
from torch.optim import Adam
from .unet import WithSinusoidalEmbedding
from tqdm import tqdm
from .util import display_images


class Model:

    def __init__(self, lr, device=None):

        # Scheduling
        self.total_timesteps = 1000
        s = 0.008
        steps = self.total_timesteps
        t = torch.linspace(0, steps, steps + 1, device=device)
        f = torch.cos((t / steps + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bar_cosine = f / f[0]
        self.beta = torch.clamp(1 - alpha_bar_cosine[1:] / alpha_bar_cosine[:-1], max=0.999).to(device)
        self.alpha = (1.0 - self.beta).to(device)
        self.alpha_bar = alpha_bar_cosine[1:].to(device)

        # Model
        self.model = WithSinusoidalEmbedding(
            input_channels=3, time_steps=self.total_timesteps
        ).to(device)

        # EMA model (we average the weights over time)
        # This is supposed to avoid falling into a local minima
        self.ema_model = WithSinusoidalEmbedding(
            input_channels=3, time_steps=self.total_timesteps
        ).to(device)

        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_decay = 0.999

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.loss = nn.HuberLoss(reduction="none")

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
                print("Loaded all")
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
        return (torch.sqrt(alpha_bar_t) * i) + (torch.sqrt(1.0 - alpha_bar_t) * noise), noise

    @torch.no_grad()
    def infer(self, data, mask, device=None):
        self.ema_model.eval()

        data_masked = data * mask
        data_heights = (data_masked[:, 0:1, :, :] * 2) - 1
        data_class = data_masked[:, 1:2, :, :]

        # x_t initialized as pure random (-1,1)
        batch_size = data.shape[0]
        x_t = randn_like(data_heights, device=device)

        alpha_bar = self.alpha_bar
        alphas = self.alpha
        betas = self.beta

        images = [data_masked[0][0].detach().to("cpu")]

        # 3. Reverse Diffusion Loop: Step from T-1 down to 0
        for t in tqdm(reversed(range(self.total_timesteps)), position=2):

            # Uncomment to display partial inference
            #if t % 100 == 0:
            #    image = x_t.to("cpu").detach()[0]
            #    images.append((image + 1) / 2)

            # Create a batch-wide timestep tensor
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Nosied input
            noised_input, _noise_added = self.noise_input_at_step(data_heights, t_tensor)

            # Incorporate noised origin back into our output
            x_t = (mask * noised_input) + (logical_not(mask) * x_t)

            # Reconstruct model input matching your train_step order
            model_input = cat([x_t, data_heights, data_class], dim=1)

            # Predict noise via the model forward pass
            inferred_noise = self.ema_model.forward(model_input, ones_like(mask), t_tensor)

            # Extract scalar coefficients for step t
            alpha_t = alphas[t].view(-1, 1, 1, 1)
            beta_t = betas[t].view(-1, 1, 1, 1)
            alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)

            # Compute mean mu_t
            noise_coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            mu_t = (1.0 / torch.sqrt(alpha_t)) * (x_t - noise_coef * inferred_noise)

            # Apply stochastic noise if we are not at the final step (t > 0)
            if t > 0:
                # Using the standard DDPM posterior variance choice
                alpha_bar_t_prev = alpha_bar[t - 1]
                sigma_t_squared = (
                    (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t) * beta_t
                )
                z = torch.randn_like(x_t, device=device)
                x_t = mu_t + torch.sqrt(sigma_t_squared) * z
            else:
                x_t = mu_t

        # 4. Invert normalization mapping back to [0, 1] range
        denoised_output = x_t
        denoised_output = (denoised_output + 1) / 2

        images.append(denoised_output.detach().to("cpu"))

        display_images(images, to_file="./infer")

        self.ema_model.train()
        return denoised_output

    def ema_update(self):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=(1 - self.ema_decay))

    def train_step(self, data, mask, expected, expected_good_mask):
        self.optimizer.zero_grad(set_to_none=True)

        data_heights = data[:, 0:1, :, :]
        data_class = data[:, 1:2, :, :]
        
        # TODO: Think about whether this helps
        # Randomly with some probability drop our classes so the model learns
        # to handle classless data with classes as supplemental 
        # 1% probability
        data_class = data_class * (torch.rand_like(data_class) > 0.01)

        # Normalize our data
        data_heights = (data_heights * 2) - 1
        expected = (expected * 2) - 1

        # Random timestep for each element in the brach
        batch_size = data.shape[0]
        timestep = torch.randint(
            0, self.total_timesteps, (batch_size,), device=data.device
        ).long()

        # Noised mask 
        alpha_bar_t = self.alpha_bar[timestep].view(-1, 1, 1, 1)
        noised_target_channel, noise_added = self.noise_input_at_step(expected, timestep)

        # combine the noised input with the masked input, order matters here (class must come last) as we onehot them in the model
        model_input = cat([noised_target_channel, data_heights, data_class], dim=1)

        # Inference, we still pass in the expected good mask so that truly
        # unknown pixels during training do not influence nearby pixels
        inferred_noise = self.model.forward(model_input, expected_good_mask, timestep)

        # Loss
        # We only care about loss in the regions we have masked but that were known good in the training input
        active_learning_region = (expected_good_mask * logical_not(mask))

        noise_loss = self.loss(inferred_noise, noise_added) * active_learning_region
        noise_loss = noise_loss.sum() / (active_learning_region.sum() + 1e-8)

        # Auxiliary loss: predict the clean image from the estimated noise
        # Gives the model a direct signal on the final output quality
        reconstructed_image = (
            noised_target_channel - torch.sqrt(1.0 - alpha_bar_t) * inferred_noise
        ) / torch.sqrt(alpha_bar_t)

        reconstructed_image_loss = self.loss(reconstructed_image, expected) * active_learning_region
        reconstructed_image_loss = reconstructed_image_loss.sum() / (active_learning_region.sum() + 1e-8)

        loss = noise_loss + reconstructed_image_loss * 0.1

        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.ema_update()

        return loss
