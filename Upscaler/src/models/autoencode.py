import random
import torch
from torch import (
    nn,
    save,
    rand,
    rand_like,
    logical_and,
    logical_not,
    transpose,
    ones_like,
    no_grad,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# from infer import display_reverse
from util import set_seed, dataloader, kl_loss, apply_batch_noise, display_images

from labml_nn.diffusion.stable_diffusion.model.autoencoder import (
    Autoencoder,
    Encoder,
    Decoder,
)

class Autoencode:

    def __init__(self, lr, device=None):
        z_channels = 512
        emb_channels = 4

        encoder = Encoder(
            in_channels=2,
            z_channels=z_channels,
            channels=32,
            channel_multipliers=[4, 2, 2],
            n_resnet_blocks=3,
        ).to(device)

        decoder = Decoder(
            out_channels=2,
            z_channels=z_channels,
            channels=32,
            channel_multipliers=[4, 2, 2],
            n_resnet_blocks=3,
        ).to(device)

        optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

        autoencoder = Autoencoder(
            encoder, decoder, emb_channels=emb_channels, z_channels=z_channels
        ).to(device)

        self.autoencoder = autoencoder
        self.optimizer = optimizer

    def load(self, file, device=None):
        try:
            print("Trying to load", file)
            if file is not None:
                checkpoint = torch.load(file, map_location=device)
                print("Loaded checkpoint")
                self.autoencoder.load_state_dict(checkpoint["autoencoder"])
                print("Loaded model")
                self.optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            print("Could not load checkpoint", e)


    def infer(self, src_frame, src_mask, device=None):
        with no_grad():
            encoded = self.autoencoder.encode((src_frame * src_mask) + (-1 * logical_not(src_mask)))
            sample = encoded.sample()
            decoded = self.autoencoder.decode(sample)

        return decoded

    def train(
        self,
        dataset,
        batch_size: int = 4,
        num_epochs: int = 150,
        seed: int = -1,
        ema_decay: float = 0.9999,
        checkpoint_path: str = None,
        device=None,
    ):
        set_seed(random.randint(0, 2**32 - 1)) if seed == -1 else set_seed(seed)
        dataset_len = len(dataset)
        dataset_per_epoch = dataset_len / batch_size
        train_loader = dataloader(dataset, batch_size)
        autoencoder = self.autoencoder
        optimizer = self.optimizer

        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=0, factor=0.1, threshold=0.001
        )

        # https://medium.com/@rahuldasari7502/building-a-beta-variational-autoencoder-%CE%B2-vae-from-scratch-with-pytorch-c5896ecc4dee suggests MSELoss(reduction=mean) can underfit
        # when used with beta-VAE
        criterion = nn.MSELoss(reduction="sum")

        def checkpoint():
            checkpoint = {
                "autoencoder": autoencoder.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save(checkpoint, checkpoint_path)
            print("Saved checkpoint to", checkpoint_path)

        for i in range(num_epochs):
            print("Optimizer state")
            for param_group in optimizer.param_groups:
                print(param_group["lr"])

            total_loss = 0
            total_kl_loss = 0
            total_output_loss = 0
            for bidx, datapoint in enumerate(
                tqdm(train_loader, desc=f"Epoch {i + 1}/{num_epochs}")
            ):
                optimizer.zero_grad(set_to_none=True)
                for_train = (
                    datapoint["terrain_with_classification"]
                    .to(device, non_blocking=True)
                    .squeeze(1)
                )
                mask = datapoint["mask"].to(device, non_blocking=True).squeeze(1)

                # print(for_train.shape, mask.shape)

                # Randomly choose to transpose the X,Y (We could during data generation rotate the entire tile before translating it to a heightmap, but that is trickier)
                if random.choice([True, False]):
                    for_train = transpose(for_train, -1, -2)
                    mask = transpose(mask, -1, -2)

                # print(for_train.shape, mask.shape)

                # Take a random element from the batch and combine its missing data with our own so that we incorporate some real loooking missing data into our own input
                batch_noise = apply_batch_noise(mask, count=random.randint(1, 2))

                # print(for_train.shape, mask.shape)

                # Generate a bunch of random numbers (batch_size,) between 0 and 1 for a known noise addition
                # Add some more noise to the image so the decoder can see some blank cells
                if random.choice([True, False, False, False, False]):
                    min_noise = 0.01
                    max_noise = 0.2
                    noise_thresh_per_batch_elt = (
                        min_noise
                        + (rand((batch_size, 1, 1, 1)) * (max_noise - min_noise))
                    ).to(device)
                    additional_noise = rand_like(for_train) > noise_thresh_per_batch_elt
                else:
                    additional_noise = ones_like(for_train)

                total_mask = logical_and(batch_noise, additional_noise)

                for_autoencoder = for_train * total_mask

                encoded = autoencoder.encode(for_autoencoder)
                sample = encoded.sample()
                decoded = autoencoder.decode(sample)
                reconstructed = (for_train * mask) + (logical_not(mask) * decoded)

                output_loss = criterion(decoded * mask, for_train * mask) / batch_size
                kl_divergence_loss = kl_loss(encoded, sample, beta=0.005)

                loss = output_loss
                if kl_divergence_loss < output_loss:
                    # print(output_loss, kl_divergence_loss)
                    loss = output_loss - kl_divergence_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_kl_loss += kl_divergence_loss.item()
                total_output_loss += output_loss.item()

                if (bidx % int(dataset_per_epoch // 100)) == 0:
                    print(
                        "Sample",
                        bidx,
                        loss.item(),
                        kl_divergence_loss.item(),
                        output_loss.item(),
                        for_train.shape[0],
                        sample.shape,
                    )

                    checkpoint()

                    for elt in range(0, for_train.shape[0]):
                        dec_for_train = for_train.to("cpu")[elt][0].detach()
                        dec_clas_train = for_train.to("cpu")[elt][1].detach()
                        dec_for_autoencoder = for_autoencoder.to("cpu")[elt][0].detach()
                        dec_decoded = decoded.to("cpu")[elt][0].detach()
                        dec_mask = mask.to("cpu")[elt][0].detach()
                        dec_reconstructed = reconstructed.to("cpu")[elt][0].detach()
                        dec_clas_reconstructed = reconstructed.to("cpu")[elt][
                            1
                        ].detach()
                        display_images(
                           [
                               dec_for_train,
                               dec_clas_train,
                               dec_mask,
                               dec_for_autoencoder,
                               dec_decoded,
                               dec_reconstructed,
                               dec_clas_reconstructed,
                           ],
                           to_file=elt,
                        )

            avg_loss = total_loss / dataset_per_epoch
            scheduler.step(avg_loss)

            print(
                f"Epoch {i + 1} | Loss {total_loss / (dataset_len / batch_size):.5f} {total_kl_loss / (dataset_len / batch_size):.5f} {total_output_loss / (dataset_len / batch_size):.5f} (Saved)"
            )

            if total_loss < 0:
                raise Exception("Explosion - self terminating")

            checkpoint()
