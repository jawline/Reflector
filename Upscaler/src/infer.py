import torch
import random
import model_loader
import matplotlib.pyplot as plt
import array
from math import inf, ceil
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
            # TODO: which_step or which_step - 1
            e = randn_like(src_frame).to(device)
            beta = sqrt(scheduler.beta[[which_step - 1]]).to(device)
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
            if sum(src_mask) > 3891:
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
            z = randn(1, 1, 128, 128)
            mask = tensor([False]).repeat(z.shape)
            images = infer_frame(
                device, z, mask, model, scheduler, num_time_steps, times
            )
            display_reverse(images)


# Compute the max distance between elements on the same row in linear time.
def compute_max_distance_row_wise(src_mask):
    rows = array.array()
    rows.extend(src_mask.shape[0])

    for y, row in enumerate(src_mask):
        rlen = len(row)
        dist_left = array.array()
        dist_right = array.array()
        results = array.array()

        dist_left.extend(rlen)
        dist_right.extend(rlen)
        results.extend(rlen)

        left_ctr = inf
        right_ctr = inf

        for x in range(rlen):
            idx_from_right = rlen - x - 1
            left_ctr = left_ctr + 1
            right_ctr = right_ctr + 1
            if row[x]:
                left_ctr = 0
            if row[idx_from_right]:
                right_ctr = 0
            dist_left[i] = left_ctr
            dist_right[idx_from_right] = right_ctr

        for x in range(rlen):
            if x == 0:
                results[x] = dist_right[x + 1]
            elif x == rlen - 1:
                results[x] = dist_left[x - 1]
            else:
                results[x] = min(dist_left[x - 1], dist_right[x + 1])

        rows[y] = results

    return tensor(rows)


def compute_distance_mask(src_mask):
    row_distances = compute_max_distances_row_wise(src_mask)
    col_distances = transpose(
        compute_max_distances_row_wise(transpose(src_mask, 0, 1)), 0, 1
    )
    return min(row_distances, col_distances)


def compute_threshold(src_mask):
    avg_dim = (src_mask.shape[0] + src_mask.shape[1]) / 2
    return ceil(avg_dim * 0.02)


# We don't want to infill everything - some regions like large bodies of water
# do no infer well and so we fill them in heuristically rather than using the model.
#
# To decide which regions to keep after infilling, we use this mask
def compute_infill_keep_mask(src_mask):
    threshold = compute_threshold(src_mask)
    return logical_and(not (src_mask), compute_max_distance(src_mask) <= threshold)


def generative_inference(
    src_frame,
    src_mask,
    kernel_width,
    kernel_height,
    device=None,
    checkpoint_path: str = None,
    num_time_steps: int = 1000,
    ema_decay: float = 0.9999,
):
    ema, scheduler = prepare_model(device, checkpoint_path, ema_decay, num_time_steps)
    model = ema.module.eval()
    times = []
    with no_grad():
        keep_mask = compute_infill_keep_mask(src_mask)
        tiles_y = ceil(src_frame.shape[0] / kernel_height)
        tiles_x = ceil(src_frame.shape[1] / kernel_width)

        y_tiles = array.array().extend(tiles_y)

        # Roughly, split the full heightmap up into chunks of kernel width and height, then for each
        # chunk compute the keep and tile masks. If the number of pixels we want to infer data for is non zero then run an inference and combine the result using the keep mask, otherwise preserve the old data and skip inference.
        for y_tile in range(tiles_y):
            print("Starting new row inference")
            x_tiles = array.array().extend(tiles_x)
            for x_tile in range(tiles_x):
                print(f"Considering inference on tile x={x_tile} y={y_tile}")
                start_y = y_tile * kernel_height
                end_y = (y_tile + 1) * kernel_height
                start_x = x_tile * kernel_width
                end_x = (x_tile + 1) * kernel_width
                tile_data = src_frame[start_y:end_y][start_x:end_x].contiguous()
                tile_mask = src_mask[start_y:end_y][start_x:end_x].contiguous()
                keep_mask = keep_mask[start_y:end_y][start_x:end_x].contiguous()

                new_tile = tile_data

                tiles_that_dont_need_inference = logical_and(src_mask, logical_not(keep_mask))
                need_to_do_inference = sum(tiles_that_dont_need_inference) != src_mask.flatten().shape[0]

                if need_to_do_inference:
                    print(f"Inferring tile x={x_tile} y={y_tile}")
                    inference = infer_frame(
                        device,
                        tile_data,
                        tile_mask,
                        model,
                        scheduler,
                        num_time_steps,
                        times,
                    )[-1]

                    # We use the keep mask here to avoid it damaging inference, since the values are None
                    # and do need prediction but we don't actually want to keep them in our result.
                    new_tile = combine_with_src_frame(src_frame, logical_not(keep_mask), result)
                else:
                    print(f"Skipping tile x={x_tile} y={y_tile}")
                x_tiles.append(new_tile)
            y_tiles.append(cat(x_tiles, dim=1)
        result = cat(y_tiles, dim=0)
        return result
