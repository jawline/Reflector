import random
import numpy as np
import torch
import model_loader
from math import isnan
from torch import nn, tensor, masked_select, randint, save
from torch.nn.functional import pad, interpolate
from torch.utils.data import DataLoader
from torch.optim import Adam
from ddpm_scheduler import DDPM_Scheduler
from unet import UNET
from tqdm import tqdm
from constants import num_time_steps


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
        num_workers=2,
        prefetch_factor=2,
    )


def norm(image, mask):
    min_value, max_value = masked_select(image.flatten(), mask.flatten()).aminmax(
        dim=-1
    )
    norm_x = (image - min_value) / (max_value - min_value)
    return norm_x, min_value, max_value


def unnorm(image, min_value, max_value):
    return (image * (max_value - min_value)) + min_value


def train(
    dataset,
    batch_size: int = 32,
    num_time_steps: int = num_time_steps,
    num_epochs: int = 150,
    seed: int = -1,
    ema_decay: float = 0.9999,
    lr=2e-5,
    checkpoint_path: str = None,
    device=None,
):
    set_seed(random.randint(0, 2**32 - 1)) if seed == -1 else set_seed(seed)
    train_loader = dataloader(dataset, batch_size)
    scheduler, model, optimizer, ema = model_loader.load(
        device, checkpoint_path, ema_decay, num_time_steps, lr
    )

    criterion = nn.MSELoss(reduction="mean")

    for i in range(num_epochs):
        total_loss = 0
        for bidx, datapoint in enumerate(
            tqdm(train_loader, desc=f"Epoch {i + 1}/{num_epochs}")
        ):
            optimizer.zero_grad()
            # TODO: emit training data in fp16 instead.
            for_train = datapoint["without_nan"].to(device)
            mask = datapoint["mask"].to(device)
            for_train, _min, _max = norm(for_train, mask)
            for_train = for_train.requires_grad_(True)
            steps = randint(0, num_time_steps, (batch_size,))
            for_train, random_data = scheduler.noise_frame(device, for_train, steps)
            output = model(for_train, steps, mask)
            loss = criterion(output * mask, random_data * mask)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)

        checkpoint = {
            "weights": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
        }

        save(checkpoint, checkpoint_path)
        print(f"Epoch {i + 1} | Loss {total_loss / (60000 / batch_size):.5f} (Saved)")
