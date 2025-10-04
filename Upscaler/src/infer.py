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
    ones,
    zeros,
)
from torch.nn.functional import pad
from constants import num_time_steps
from torchvision.transforms.v2 import Resize
from torchvision.transforms.functional import adjust_sharpness
import preprocess


def aa(tensor):
    resize1 = Resize(size=(tensor.shape[-2] * 2, tensor.shape[-1] * 2), antialias=True)
    resize2 = Resize(size=(tensor.shape[-2], tensor.shape[-1]), antialias=True)
    sharpened = adjust_sharpness(tensor, sharpness_factor=2)
    return resize2.transform(resize1.transform(sharpened, {}), {})


def display_reverse(images: List, to_file=None):
    fig, axes = plt.subplots(1, len(images), figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i]
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


def infer_frame(device, src_frame, src_mask, model):
    with no_grad():
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
min_tile_overlap = 64


def tiled_inference(src_frame, src_mask, src_keep, autoencoder, kernel_size, device):
    src_frame = src_frame.to(device)
    src_mask = src_mask.to(device)
    src_keep = src_keep.to(device)

    src_height = src_frame.shape[-2]
    src_width = src_frame.shape[-1]

    print("Beginning tiled inference", src_frame.shape, src_mask.shape, src_keep.shape)

    for start_y in range(0, src_frame.shape[-2], kernel_size - min_tile_overlap):
        print("Start y", start_y)
        for start_x in range(0, src_frame.shape[-1], kernel_size - min_tile_overlap):
            last_x = False
            last_y = False

            if start_x + kernel_size > src_width:
                start_x -= start_x + kernel_size - src_width
                last_x = True

            if start_y + kernel_size > src_height:
                start_y -= start_y + kernel_size - src_height
                last_y = True

            end_x = start_x + kernel_size
            end_y = start_y + kernel_size

            tile_data = src_frame[:, :, start_y:end_y, start_x:end_x]
            tile_mask = src_mask[:, :, start_y:end_y, start_x:end_x]
            tile_keep = src_keep[:, :, start_y:end_y, start_x:end_x]

            print(
                "Extracted tile shape",
                start_x,
                start_y,
                end_x,
                end_y,
                end_x - start_x,
                end_y - start_y,
                tile_data.shape,
            )

            tiles_that_dont_need_inference = logical_and(
                tile_mask, logical_not(tile_keep)
            )

            need_to_do_inference = (
                sum(tiles_that_dont_need_inference) != src_mask.flatten().shape[0]
            )

            if need_to_do_inference:
                print("Inferring tile", start_x, start_y, tile_data.shape)

                # TODO: The model has overfitted on seeing some noise so this is necessary to produce sane outputs. Remove the rand in training
                #additional_tile_mask = logical_and(
                #    tile_mask, rand_like(tile_data) > 0.1
                #)
                #additional_tile_data = (tile_data * additional_tile_mask) + (
                #    -1 * logical_not(additional_tile_mask)
                #)

                inference = infer_frame(
                    device,
                    tile_data,
                    tile_mask,
                    autoencoder,
                )
                # display_reverse(
                #    [
                #        additional_tile_data.to("cpu"),
                #        additional_tile_mask.to("cpu"),
                #        inference.to("cpu"),
                #        ])

                # Leave a little unpredicted margin at the edge of the frame to be inferred by the next iteration which will have a better picture due to the overlap.
                # In general it seems best not to keep inferences too close to the edge as they will be better served by an overlapping subsequent inference.
                write_back_start_x = start_x
                write_back_end_x = end_x
                if not last_x:
                    write_back_end_x -= min_tile_overlap // 2

                write_back_start_y = start_y
                write_back_end_y = end_y
                if not last_y:
                    write_back_end_y -= min_tile_overlap // 2

                print(
                    "Write backs",
                    write_back_start_x,
                    write_back_end_x,
                    write_back_start_y,
                    write_back_end_y,
                )

                # Only keep the regions we actually inferred
                reconstructed_frame = (tile_data * logical_not(tile_keep)) + (
                    inference * tile_keep
                )
                print("Reconstructed shape", reconstructed_frame.shape)

                write_back_height = write_back_end_y - write_back_start_y
                write_back_width = write_back_end_x - write_back_start_x

                reconstructed_frame = reconstructed_frame[
                    :,
                    :,
                    : write_back_height,
                    : write_back_width,
                ]

                src_frame[
                    :,
                    :,
                    write_back_start_y:write_back_end_y,
                    write_back_start_x:write_back_end_x,
                ] = reconstructed_frame

                src_mask[
                    :,
                    :,
                    write_back_start_y:write_back_end_y,
                    write_back_start_x:write_back_end_x,
                ] = ones((1, 1, write_back_height, write_back_width), dtype=torch.bool)

                src_keep[
                    :,
                    :,
                    write_back_start_y:write_back_end_y,
                    write_back_start_x:write_back_end_x,
                ] = zeros((1, 1, write_back_height, write_back_width), dtype=torch.bool)

                # display_reverse(
                #    [
                #        tile_data.to("cpu"),
                #        tile_mask.to("cpu"),
                #        inference.to("cpu"),
                #        tile_keep.to("cpu"),
                #        reconstructed_frame.to("cpu"),
                #        src_frame.to("cpu"),
                #        src_mask.to("cpu"),
                #    ]
                # )
            else:
                print("Skipping tile", start_x, start_y)

    return src_frame


def whole_datasource_tiled_inference(
    src_frame,
    src_mask,
    kernel_size,
    device=None,
    checkpoint_path: str = None,
    num_time_steps: int = num_time_steps,
    ema_decay: float = 0.9999,
):
    autoencoder, _optimizer = model_loader.load_autoencoder(
        device, checkpoint_path, 0.1
    )

    print("Starting to compute infill mask", src_mask.shape)
    keep_mask = compute_infill_keep_mask(src_mask[0][0]).reshape(src_mask.shape)
    print("Computed infill mask")

    result = tiled_inference(
        src_frame.clone(), src_mask, keep_mask, autoencoder, kernel_size, device
    )
    print("Inf result", result.shape)

    # result = aa(result)
    # print("AA result", result.shape)

    # display_reverse(
    #    [
    #        src_frame.to("cpu"),
    #        src_mask.to("cpu"),
    #        keep_mask.to("cpu"),
    #        result.to("cpu"),
    #    ]
    # )

    return result
