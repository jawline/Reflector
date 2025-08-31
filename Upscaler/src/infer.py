import torch
import random
import model_loader
import matplotlib.pyplot as plt
from array import array
from math import inf, ceil
from typing import List
from timm.utils import ModelEmaV3
from unet import UNET
from ddpm_scheduler import DDPM_Scheduler
from einops import rearrange
from tqdm import tqdm
from torch import (
    cat,
    tensor,
    no_grad,
    logical_and,
    logical_or,
    logical_not,
    rand,
    rand_like,
    randn,
    randn_like,
    sqrt,
    sum,
    transpose,
)
from torch.nn.functional import pad
from constants import num_time_steps, tile_width, tile_height
import preprocess


def display_reverse(images: List, to_file=None):
    fig, axes = plt.subplots(1, len(images), figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, "c h w -> h w c")
        x = x.numpy()
        ax.imshow(x, vmin=0, vmax=1)
        ax.axis("off")

    if to_file is not None:
        path = f"{to_file}.png"
        plt.savefig(path, dpi=1200)
        plt.close()
        print("Rendered PNG", path)
    else:
        plt.show()


def prepare_model(device, checkpoint_path, ema_decay, num_time_steps):
    scheduler, model, optimizer, ema, scaler = model_loader.load(
        device, checkpoint_path, ema_decay, num_time_steps, lr=0.1
    )
    return ema, scheduler


def step(device, frame, model, scheduler, step):
    predicted_noise = model(frame, [step])
    return scheduler.denoise_from(device, frame, predicted_noise, step)


def infer_noise_mask(device, scheduler, src_frame, src_mask, step):
    noised_frame, _random_data = scheduler.noise_frame(src_frame, [step])
    return noised_frame * src_mask


def combine_with_src_frame(src_frame, src_mask, frame):
    return (src_frame * src_mask) + (frame * logical_not(src_mask))


def infer_frame(device, src_frame, src_mask, model, num_time_steps, sample_times):
    src_frame = src_frame.to(device)
    src_mask = src_mask.to(device)

    encoded = model.encode((src_frame * src_mask) + (-1 * logical_not(src_mask)))
    sample = encoded.sample()
    decoded = model.decode(sample)

    # TODO: Exclusively use the predicted frame as an option rather than combining it with the origin frame

    # display_reverse([src_frame.to("cpu"), a_little_noise.to("cpu"), src_mask.to("cpu"), decoded.to("cpu")])

    return decoded


def masked_inference(
    dataset,
    device=None,
    checkpoint_path: str = None,
    num_time_steps: int = num_time_steps,
    ema_decay: float = 0.9999,
):
    ema, scheduler = prepare_model(device, checkpoint_path, ema_decay, num_time_steps)
    times = []
    with no_grad():
        for elt in dataset:
            model = ema.module.eval()
            datapoint = elt

            src_frame = datapoint["without_nan"].reshape(
                (1, 1, tile_width, tile_height)
            )
            src_mask = datapoint["mask"].reshape((1, 1, tile_width, tile_height))
            display_reverse([src_frame.to("cpu"), src_frame.to("cpu")])

            # Skip a candidate if it has a lot of data
            if sum(src_mask) > (tile_width * tile_height) * 0.96:
                print("Skip candidate, too nice")
                continue

            images = infer_frame(
                device, src_frame, src_mask, model, scheduler, num_time_steps, times
            )

            display_reverse(images)


def generative_inference(
    device=None,
    checkpoint_path: str = None,
    num_time_steps: int = num_time_steps,
    ema_decay: float = 0.9999,
):
    ema, scheduler = prepare_model(device, checkpoint_path, ema_decay, num_time_steps)
    times = [100, 200, 300, 400, 500, 600, 700, 800, 900]

    with no_grad():
        model = ema.module.eval()
        for i in range(10):
            z = rand(size=(1, 1, tile_width, tile_height))
            mask = tensor([False]).repeat(z.shape)
            images = infer_frame(
                device, z, mask, model, scheduler, num_time_steps, times
            )
            display_reverse(images)


# Compute the max distance between elements on the same row in linear time.
def compute_max_distance_row_wise(src_mask):
    rows = []

    for y, row in enumerate(src_mask):
        rlen = len(row)
        dist_left = array("f")
        dist_right = array("f")
        results = array("f")

        dist_left.extend([0] * rlen)
        dist_right.extend([0] * rlen)
        results.extend([0] * rlen)

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
            dist_left[x] = left_ctr
            dist_right[idx_from_right] = right_ctr

        for x in range(rlen):
            if x == 0:
                results[x] = dist_right[x + 1]
            elif x == rlen - 1:
                results[x] = dist_left[x - 1]
            else:
                results[x] = min(dist_left[x - 1], dist_right[x + 1])

        rows.append(results)

    return tensor(rows)


def compute_distance_mask(src_mask):
    row_distances = compute_max_distance_row_wise(src_mask)
    col_distances = transpose(
        compute_max_distance_row_wise(transpose(src_mask, 0, 1)), 0, 1
    )
    return torch.min(row_distances, col_distances)


def compute_threshold(src_mask):
    avg_dim = (src_mask.shape[0] + src_mask.shape[1]) / 2
    return ceil(avg_dim * 0.1)


# We don't want to infill everything - some regions like large bodies of water
# do no infer well and so we fill them in heuristically rather than using the model.
#
# To decide which regions to keep after infilling, we use this mask
def compute_infill_keep_mask(src_mask):
    threshold = compute_threshold(src_mask)
    return logical_and(
        logical_not(src_mask), compute_distance_mask(src_mask) <= threshold
    )


# To improve output we make sure we overlap some of the rows and columns with previous inferences during the tiled inference
tile_overlap = 0


def whole_datasource_tiled_inference(
    src_frame,
    src_mask,
    kernel_width,
    kernel_height,
    device=None,
    checkpoint_path: str = None,
    num_time_steps: int = num_time_steps,
    ema_decay: float = 0.9999,
):
    autoencoder, _optimizer = model_loader.load_autoencoder(
        device, checkpoint_path, 0.1
    )
    times = []

    with no_grad():
        print("Starting to compute infill mask")
        print(src_mask.shape)
        # keep_mask = compute_infill_keep_mask(src_mask[0][0]).reshape(src_mask.shape)
        keep_mask = torch.ones(src_frame.shape) * logical_not(src_mask)

        print("Computed infill mask")

        y_tiles = []

        # Roughly, split the full heightmap up into chunks of kernel width and height, then for each
        # chunk compute the keep and tile masks. If the number of pixels we want to infer data for is non zero then run an inference and combine the result using the keep mask, otherwise preserve the old data and skip inference.
        for start_y in range(0, src_frame.shape[-2], kernel_height - tile_overlap):
            print("Starting new row inference")
            x_tiles = []

            for start_x in range(0, src_frame.shape[-1], kernel_width - tile_overlap):
                # TODO: Rather than this we could combine all the images into a tensor and then use chunks
                end_y = start_y + kernel_height
                end_x = start_x + kernel_width

                print(
                    f"Starting part shapes {src_frame.shape} {keep_mask.shape} {start_x} {end_x} {start_y} {end_y}"
                )

                # Whenever possible, we pad with real data from earlier in the frame rather than unknown values to make the
                # model more stable.
                pad_y = max(kernel_height - (src_frame.shape[-2] - start_y), 0)
                pad_x = max(kernel_width - (src_frame.shape[-1] - start_x), 0)

                tile_data = src_frame[:, :, start_y - pad_y :end_y, start_x - pad_x :end_x].contiguous()

                tile_mask = (
                    src_mask[:, :, start_y - pad_y :end_y, start_x - pad_x :end_x].contiguous().to(device)
                )

                tile_keep_mask = keep_mask[
                    :, :, start_y - pad_y :end_y, start_x - pad_x :end_x
                ].contiguous()

                print(
                    f"extracted tiles {tile_data.shape} {tile_mask.shape} {tile_keep_mask.shape}"
                )

                tile_data = tile_data.to(device)
                tile_mask = tile_mask.to(device)
                tile_keep_mask = tile_keep_mask.to(device)

                new_tile = tile_data
                print(
                    f"padded tiles {tile_data.shape}, {tile_mask.shape}, {tile_keep_mask.shape}"
                )

                tiles_that_dont_need_inference = logical_and(
                    tile_mask, logical_not(tile_keep_mask)
                )

                need_to_do_inference = (
                    sum(tiles_that_dont_need_inference) != src_mask.flatten().shape[0]
                )

                if need_to_do_inference:
                    print(f"Inferring tile {tile_data.shape} {tile_mask.shape}")

                    inference = infer_frame(
                        device,
                        tile_data.reshape((1, 1, kernel_width, kernel_height)),
                        tile_mask.reshape((1, 1, kernel_width, kernel_height)),
                        autoencoder,
                        num_time_steps,
                        times,
                    )

                    print("inference", inference.shape)

                    # We use the keep mask here to avoid it damaging inference, since the values are None
                    # and do need prediction but we don't actually want to keep them in our result.
                    print(
                        "Pre combine",
                        tile_data.shape,
                        tile_keep_mask.shape,
                        inference.shape,
                    )
                    new_tile = combine_with_src_frame(
                        tile_data, logical_not(tile_keep_mask), inference
                    )
                    print("Tile shape", new_tile.shape)
                else:
                    print(f"Skipping tile")

                # Remove the tile overlap
                if start_x != 0:
                    new_tile = new_tile[:, :, :, tile_overlap:]

                if start_y != 0:
                    new_tile = new_tile[:, :, tile_overlap:, :]



                # Pad back into kernel size
                new_tile = pad(new_tile[:,:,pad_y:,pad_x:], (0, pad_x, 0, pad_y), "constant", 0.0)

                print(new_tile.shape)
                print("tile", new_tile.shape)
                x_tiles.append(new_tile.squeeze(0).squeeze(0))

            y_tiles.append(cat(x_tiles, dim=1))

        src_shape = src_frame.shape
        print("tile shape", y_tiles[0].shape)
        result = cat(y_tiles, dim=0).unsqueeze(0).unsqueeze(0)
        print("result pre trunc", result.shape)
        result = result[:, :, 0 : src_frame.shape[-2], 0 : src_frame.shape[-1]]
        print("post trunc", result.shape)

        display_reverse(
            [
                src_frame.reshape(src_shape).to("cpu"),
                src_mask.reshape(src_shape).to("cpu"),
                keep_mask.reshape(src_shape).to("cpu"),
                result.reshape(src_shape).to("cpu"),
            ]
        )

        return result
