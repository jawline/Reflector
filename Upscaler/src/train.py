import random
import numpy as np
import torch
import model_loader
from math import isnan
from torch import (
    nn,
    tensor,
    masked_select,
    randint,
    save,
    cat,
    rand,
    rand_like,
    randn_like,
    logical_not,
    logical_or,
    logical_and,
    sqrt,
    transpose
)
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn.functional import pad, interpolate
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ddpm_scheduler import DDPM_Scheduler
from unet import UNET
from tqdm import tqdm
from constants import num_time_steps
from infer import display_reverse
from preprocess import unnorm


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=6,
        prefetch_factor=64,
    )


def kl_loss(encoded_distribution, sample, *, beta=None):
    q_dist = Normal(
        encoded_distribution.mean, torch.exp(0.5 * encoded_distribution.log_var)
    )
    p_dist = Normal(
        torch.zeros_like(encoded_distribution.mean),
        torch.ones_like(encoded_distribution.log_var),
    )

    sample = encoded_distribution.sample()
    log_qdist = q_dist.log_prob(sample)
    log_pdist = p_dist.log_prob(sample)
    kl_div = log_qdist - log_pdist

    # print(kl_div.shape)

    kl_div = kl_div.sum(-1).mean()
    # print(kl_div.item())

    return kl_div * beta


def train_auto(
    dataset,
    batch_size: int = 8,
    num_time_steps: int = num_time_steps,
    num_epochs: int = 150,
    seed: int = -1,
    ema_decay: float = 0.9999,
    lr=0.001,
    checkpoint_path: str = None,
    device=None,
):
    set_seed(random.randint(0, 2**32 - 1)) if seed == -1 else set_seed(seed)
    dataset_len = len(dataset)
    dataset_per_epoch = dataset_len / batch_size
    train_loader = dataloader(dataset, batch_size)
    autoencoder, optimizer = model_loader.load_autoencoder(device, checkpoint_path, lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1, threshold=0.001
    )

    # https://medium.com/@rahuldasari7502/building-a-beta-variational-autoencoder-%CE%B2-vae-from-scratch-with-pytorch-c5896ecc4dee suggests MSELoss(reduction=mean) can underfit
    # when used with beta-VAE
    criterion = nn.MSELoss(reduction="sum")

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
            for_train = datapoint["without_nan"].to(device, non_blocking=True)
            mask = datapoint["mask"].to(device, non_blocking=True)

            # Randomly choose to transpose the X,Y (We could during data generation rotate the entire tile before translating it to a heightmap, but that is trickier)
            if random.choice([True, False]):
                for_train = transpose(for_train, -1, -2)
                mask = transpose(mask, -1, -2)

            # Generate a bunch of randon numbers (batch_size,) between 0 and 1 for a known noise addition
            # Add some more noise to the image so the decoder can see some blank cells
            batch_noise = (0.01 + (rand((batch_size, 1, 1, 1)) * 0.8)).to(device) # Between 1% and 81% total noise

            additional_noise = rand_like(for_train) > batch_noise
            total_mask = logical_and(mask, additional_noise)

            for_autoencoder = (for_train * total_mask) + (
                randn_like(for_train) * logical_not(total_mask)
            )

            encoded = autoencoder.encode(for_autoencoder)
            sample = encoded.sample()
            decoded = autoencoder.decode(sample)
            reconstructed = (for_train * mask) + (logical_not(mask) * decoded)

            # We are using MSEloss
            output_loss = criterion(decoded * mask, for_train * mask) / batch_size
            kl_divergence_loss = kl_loss(encoded, sample, beta=0.0001)

            loss = output_loss
            if kl_divergence_loss < output_loss:
                #print(output_loss, kl_divergence_loss)
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
                    sample.shape,
                )

                for_train = for_train.to("cpu")[0].unsqueeze(0).detach()
                for_autoencoder = for_autoencoder.to("cpu")[0].unsqueeze(0).detach()
                decoded = decoded.to("cpu")[0].unsqueeze(0).detach()
                mask = mask.to("cpu")[0].unsqueeze(0).detach()
                reconstructed = reconstructed.to("cpu")[0].unsqueeze(0).detach()
                display_reverse([
                 for_train,
                 mask,
                 for_autoencoder,
                 decoded,
                 reconstructed
                ], to_file=True)

        avg_loss = total_loss / dataset_per_epoch
        scheduler.step(avg_loss)

        print(
            f"Epoch {i + 1} | Loss {total_loss / (dataset_len / batch_size):.5f} {total_kl_loss / (dataset_len / batch_size):.5f} {total_output_loss / (dataset_len / batch_size):.5f} (Saved)"
        )

        if total_loss < 0:
            raise Exception("Explosion - self terminating")

        checkpoint = {
            "autoencoder": autoencoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        save(checkpoint, checkpoint_path)


def train(
    dataset,
    batch_size: int = 8,
    num_time_steps: int = num_time_steps,
    num_epochs: int = 150,
    seed: int = -1,
    ema_decay: float = 0.9999,
    lr=0.001,
    checkpoint_path: str = None,
    device=None,
):
    set_seed(random.randint(0, 2**32 - 1)) if seed == -1 else set_seed(seed)
    dataset_len = len(dataset)
    train_loader = dataloader(dataset, batch_size)
    scheduler, model, optimizer, ema, scaler = model_loader.load(
        device, checkpoint_path, ema_decay, num_time_steps, lr
    )

    criterion = nn.MSELoss(reduction="mean")

    for i in range(num_epochs):
        total_loss = 0
        for bidx, datapoint in enumerate(
            tqdm(train_loader, desc=f"Epoch {i + 1}/{num_epochs}")
        ):
            for_train = datapoint["without_nan"].to(device, non_blocking=True)
            mask = datapoint["mask"].to(device, non_blocking=True)

            steps = randint(0, num_time_steps, (batch_size,), device=device)

            for_train_random_substitutions = randn_like(for_train, device=device)

            # Replace the masked regions with noise so they aren't constants
            for_train = (for_train * mask) + (
                for_train_random_substitutions
                * sqrt(scheduler.beta[steps].view(len(steps), 1, 1, 1))
                * logical_not(mask)
            )

            # print(steps, scheduler.beta[steps], sqrt(scheduler.beta[steps]))
            # display_reverse([for_train[0].reshape((1, 1, 256, 256)).to("cpu"), for_train_new[0].reshape((1, 1, 256, 256)).to("cpu"), for_train_random_substitutions[0].reshape((1, 1, 256, 256)).to("cpu")])

            for_train, random_data = scheduler.noise_frame(for_train, steps)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(for_train, steps)
                loss = criterion(output * mask, random_data * mask)

            total_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

        checkpoint = {
            "weights": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "scaler": scaler.state_dict(),
        }

        save(checkpoint, checkpoint_path)
        print(
            f"Epoch {i + 1} | Loss {total_loss / (dataset_len / batch_size):.5f} (Saved)"
        )
