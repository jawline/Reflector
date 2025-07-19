import torch
from typing import List
from timm.utils import ModelEmaV3
from unet import UNET
from ddpm_scheduler import DDPM_Scheduler
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt


def display_reverse(images: List):
    fig, axes = plt.subplots(1, len(images), figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, "c h w -> h w c")
        x = x.numpy()
        ax.imshow(x)
        ax.axis("off")
    plt.show()


def prepare_model(checkpoint_path, ema_decay, num_time_steps):
    checkpoint = torch.load(checkpoint_path)
    model = UNET()
    model.load_state_dict(checkpoint["weights"])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint["ema"])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    return ema, scheduler


def infer_frame(src_frame, src_mask, model, scheduler, num_time_steps, sample_times):
    images = []

    for t in tqdm(
        reversed(range(1, num_time_steps)), desc=f"Infer Step {num_time_steps}"
    ):
        t = [t]
        temp = scheduler.beta[t] / (
            (torch.sqrt(1 - scheduler.alpha[t])) * (torch.sqrt(1 - scheduler.beta[t]))
        )
        src_frame = (1 / (torch.sqrt(1 - scheduler.beta[t]))) * src_frame - (
            temp
            * model(
                src_frame, t, src_frame
            ).cpu()  # The third argument is a mask which is ignored with no_grad
        )
        if t[0] in sample_times:
            images.append(src_frame)
        e = torch.randn(1, 1, 64, 64)
        src_frame = src_frame + (e * torch.sqrt(scheduler.beta[t]))

    temp = scheduler.beta[0] / (
        (torch.sqrt(1 - scheduler.alpha[0])) * (torch.sqrt(1 - scheduler.beta[0]))
    )

    x = (1 / (torch.sqrt(1 - scheduler.beta[0]))) * src_frame - (
        temp
        * model(
            src_frame, [0], src_frame
        ).cpu()  # The third argument is a mask which is ignored with no_grad
    )

    images.append(x)
    return images


def generative_inference(
    checkpoint_path: str = None,
    num_time_steps: int = 1000,
    ema_decay: float = 0.9999,
):
    ema, scheduler = prepare_model(checkpoint_path, ema_decay, num_time_steps)
    times = [0, 5, 15, 50, 100, 200, 300, 400, 550, 700, 999]

    with torch.no_grad():
        model = ema.module.eval()
        for i in range(10):
            z = torch.randn(1, 1, 64, 64)
            images = infer_frame(z, z, model, scheduler, num_time_steps, times)
            display_reverse(images)
