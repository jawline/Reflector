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
    randn_like,
    logical_not,
    logical_or,
    logical_and,
    sqrt,
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
from dataset import norm
from infer import display_reverse


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


def kl_loss(encoded_distribution, *, beta=None):
    q_dist = Normal(
        encoded_distribution.mean, torch.exp(0.5 * encoded_distribution.log_var)
    )
    p_dist = Normal(
        torch.zeros_like(encoded_distribution.mean),
        torch.ones_like(encoded_distribution.log_var),
    )
    return kl_divergence(q_dist, p_dist).sum() * beta


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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.1, threshold=0.001)

    # https://medium.com/@rahuldasari7502/building-a-beta-variational-autoencoder-%CE%B2-vae-from-scratch-with-pytorch-c5896ecc4dee suggests MSELoss(reduction=mean) can underfit
    # when used with beta-VAE
    criterion = nn.MSELoss(reduction="sum")

    for i in range(num_epochs):
        total_loss = 0
        total_kl_loss = 0
        total_output_loss = 0
        for bidx, datapoint in enumerate(
            tqdm(train_loader, desc=f"Epoch {i + 1}/{num_epochs}")
        ):
            optimizer.zero_grad(set_to_none=True)
            for_train = datapoint["without_nan"].to(device, non_blocking=True)
            mask = datapoint["mask"].to(device, non_blocking=True)

            encoded = autoencoder.encode(for_train * mask)
            decoded = autoencoder.decode(encoded.sample())
            # We are using MSEloss 
            output_loss = criterion(decoded * mask, for_train * mask) / batch_size
            kl_divergence_loss = kl_loss(encoded, beta=0.00001) / batch_size

            loss = output_loss + kl_divergence_loss
            loss.backward()
            optimizer.step()

            if (bidx % int(dataset_per_epoch // 100)) == 0:
                print("Sample", bidx, kl_divergence_loss.item(), output_loss.item())

            total_loss += loss.item()
            total_kl_loss += kl_divergence_loss.item()
            total_output_loss += output_loss.item()

            #display_reverse([
            #    for_train.to("cpu")[0].unsqueeze(0).detach(),
            #    decoded.to("cpu")[0].unsqueeze(0).detach(),
            #])

        avg_loss = total_loss / dataset_per_epoch
        scheduler.step(avg_loss)
        

        print(
            f"Epoch {i + 1} | Loss {total_loss / (dataset_len / batch_size):.5f} {total_kl_loss / (dataset_len / batch_size):.5f} {total_output_loss / (dataset_len / batch_size):.5f} (Saved)"
        )

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
