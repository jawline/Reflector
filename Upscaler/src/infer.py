import torch
import random
import model_loader
import matplotlib.pyplot as plt
from typing import List
from timm.utils import ModelEmaV3
from unet import UNET
from ddpm_scheduler import DDPM_Scheduler
from einops import rearrange
from tqdm import tqdm
from torch import no_grad, logical_not, randn, randn_like, sqrt, sum


def display_reverse(images: List):
    fig, axes = plt.subplots(1, len(images), figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, "c h w -> h w c")
        x = x.numpy()
        ax.imshow(x, vmin=0, vmax=1)
        ax.axis("off")
    plt.show()


def prepare_model(device, checkpoint_path, ema_decay, num_time_steps):
    scheduler, model, optimizer, ema = model_loader.load(
        device, checkpoint_path, ema_decay, num_time_steps, lr=0.1
    )
    return ema, scheduler


def step(device, frame, model, scheduler, step):
    # The third argument to model is a mask which is ignored with no_grad
    predicted_noise = model(frame, [step], frame)
    return scheduler.denoise_from(device, frame, predicted_noise, step)


def infer_noise_mask(device, scheduler, src_frame, src_mask, step):
    noised_frame, _random_data = scheduler.noise_frame(device, src_frame, [step])
    return noised_frame * src_mask


def combine_with_src_frame(src_frame, src_mask, frame):
    return (src_frame * src_mask) + (frame * logical_not(src_mask))


def infer_frame(
    device, src_frame, src_mask, model, scheduler, num_time_steps, sample_times
):
    images = []

    images.append(src_frame * src_mask)

    src_frame = src_frame.to(device)
    src_mask = src_mask.to(device)

    guess_mask = (logical_not(src_mask)).to(device)
    masked_src_frame = (src_frame * src_mask) + (
        randn_like(src_frame).to(device) * guess_mask
    )

    images.append(masked_src_frame.to("cpu"))

    for which_step in tqdm(
        reversed(range(0, num_time_steps)), desc=f"Infer Step {num_time_steps}"
    ):
        masked_src_frame = combine_with_src_frame(
            infer_noise_mask(device, scheduler, src_frame, src_mask, which_step),
            src_mask,
            masked_src_frame,
        )
        masked_src_frame = step(device, masked_src_frame, model, scheduler, which_step)

        # This might not be necessary, but per the article I constructed this from
        # "For example, estimating and subtracting the total amount of noise in the beginning
        # of the iterative process all at once leads to very incoherent samples, so in practice
        # adding a bit of the noise back and iterating through every time step has empirically
        # been shown to generate better samples."
        if which_step != 0:
            e = randn_like(src_frame).to(device)
            beta = sqrt(scheduler.beta[[which_step]]).to(device)
            masked_src_frame = masked_src_frame + (e * beta)

        masked_src_frame = (masked_src_frame * guess_mask) + (src_frame * src_mask)
        if which_step in sample_times:
            images.append(masked_src_frame.to("cpu"))

    masked_src_frame = step(device, masked_src_frame.to(device), model, scheduler, 0)
    masked_src_frame = combine_with_src_frame(src_frame, src_mask, masked_src_frame)

    images.append(masked_src_frame.to("cpu"))
    return images


def masked_inference(
    dataset,
    device=None,
    checkpoint_path: str = None,
    num_time_steps: int = 50,
    ema_decay: float = 0.9999,
):
    ema, scheduler = prepare_model(device, checkpoint_path, ema_decay, num_time_steps)
    times = []
    with no_grad():
        while True:
            model = ema.module.eval()
            datapoint = random.choice(dataset)

            src_frame = datapoint["without_nan"]
            src_mask = datapoint["mask"]

            # Skip a candidate if it has a lot of data
            if sum(src_mask) > 3686:
                print("Skip candidate, too nice")
                continue

            images = infer_frame(
                device, src_frame, src_mask, model, scheduler, num_time_steps, times
            )
            display_reverse(images)


def generative_inference(
    device=None,
    checkpoint_path: str = None,
    num_time_steps: int = 1000,
    ema_decay: float = 0.9999,
):
    ema, scheduler = prepare_model(device, checkpoint_path, ema_decay, num_time_steps)
    times = []

    with no_grad():
        model = ema.module.eval()
        for i in range(10):
            z = randn(1, 1, 64, 64)
            images = infer_frame(
                device, z, z < 0.1, model, scheduler, num_time_steps, times
            )
            display_reverse(images)
