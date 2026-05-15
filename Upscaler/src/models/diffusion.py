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
        self.beta = torch.linspace(1e-4, 0.02, self.total_timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Model
        self.model = WithSinusoidalEmbedding(
            input_channels=3, time_steps=self.total_timesteps
        ).to(device)
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
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                print("Loaded all")
        except Exception as e:
            print("Could not load checkpoint", e)

    def checkpoint(self, checkpoint_path):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        print("Saving checkpoint to", checkpoint_path)
        save(checkpoint, checkpoint_path)
        print("Saved checkpoint to", checkpoint_path)

    @torch.no_grad()
    def infer(self, data, mask, device=None):
        self.model.eval()

        # 1. Isolate and normalize known conditioning channels
        data_masked = data * mask
        data_heights = (data_masked[:, 0:1, :, :] * 2) - 1
        data_class = data_masked[:, 1:2, :, :]

        # 2. Initialize x_T as pure Gaussian noise matching height dimensions
        batch_size = data.shape[0]
        x_t = randn_like(data_heights, device=device)

        alpha_bar = self.alpha_bar.to(device)
        alphas = self.alpha.to(device)
        betas = self.beta.to(device)

        images = [data_masked[0][0].detach().to("cpu")]

        # 3. Reverse Diffusion Loop: Step from T-1 down to 0
        for t in tqdm(reversed(range(self.total_timesteps)), position=2):

            if t % 100 == 0:
                image = x_t.to("cpu").detach()[0]
                images.append(image)

            # Create a batch-wide timestep tensor
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Reconstruct model input matching your train_step order
            model_input = cat([x_t, data_heights, data_class], dim=1)

            # Predict noise via the model forward pass
            inferred_noise = self.model.forward(model_input, ones_like(mask), t_tensor)

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

        self.model.train()
        return denoised_output

    def train_step(self, data, mask, expected, expected_good_mask, device=None):
        self.optimizer.zero_grad(set_to_none=True)

        data = data * mask

        data_heights = data[:, 0:1, :, :]
        data_class = data[:, 1:2, :, :]

        # Normalize our data
        data_heights = (data_heights * 2) - 1
        expected = (expected * 2) - 1

        # Random timestep for each element in the brach
        batch_size = data.shape[0]
        timestep = torch.randint(
            0, self.total_timesteps, (batch_size,), device=device
        ).long()

        # Noise to add to the heights on a separate channel
        noise = torch.randn_like(data_heights)

        # We noise the expected outcome since the model is learning to predict
        # the difference from the noise at this step and the expected outcome
        # given the rest of the data.
        alpha_bar_t = self.alpha_bar[timestep].view(-1, 1, 1, 1)
        noised_target_channel = (
            torch.sqrt(alpha_bar_t) * expected + torch.sqrt(1.0 - alpha_bar_t) * noise
        )

        # combine the noised input with the masked input, order matters here (class must come last) as we onehot them in the model
        model_input = cat([noised_target_channel, data_heights, data_class], dim=1)

        # Inference, we still pass in the expected good mask so that truly
        # unknown pixels during training do not influence nearby pixels
        inferred_noise = self.model.forward(model_input, expected_good_mask, timestep)


        # Loss
        # We only care about loss in the regions we have masked but that were known good in the training input
        active_learning_region = (expected_good_mask * logical_not(mask)).float()

        loss = self.loss(inferred_noise, noise) * active_learning_region
        loss = loss.sum() / (active_learning_region.sum() + 1e-8)

        loss.backward()

        self.optimizer.step()

        denoised_target = (
            noised_target_channel - torch.sqrt(1.0 - alpha_bar_t) * inferred_noise
        ) / torch.sqrt(alpha_bar_t)

        display_images(
            [
                model_input[0][0].detach().to("cpu"),
                model_input[0][1].detach().to("cpu"),
                model_input[0][2].detach().to("cpu"),
                active_learning_region.detach().to("cpu"),
                denoised_target
            ],
            to_file="./train",
        )

        return (denoised_target + 1) / 2, loss
