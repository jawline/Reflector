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
    logical_not,
    logical_or,
    logical_and,
)
from torch.nn.functional import pad, interpolate
from torch.utils.data import DataLoader
from torch.optim import Adam
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


def train(
    dataset,
    batch_size: int = 4,
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

            for_train, random_data = scheduler.noise_frame(
                for_train, steps
            )

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(for_train, steps)
                loss = criterion(
                    output * mask, random_data * mask 
                )

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
