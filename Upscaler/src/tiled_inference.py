import heapq
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt
from math import ceil
from typing import List
from torch import (
    tensor,
    logical_and,
    logical_not,
    sum,
    ones,
    zeros,
)
from constants import num_time_steps
from torchvision.transforms.v2 import Resize
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import adjust_sharpness
from models.util import display_images


def aa(tensor):
    resize1 = Resize(size=(tensor.shape[-2] * 2, tensor.shape[-1] * 2), antialias=True)
    resize2 = Resize(size=(tensor.shape[-2], tensor.shape[-1]), antialias=True)
    sharpened = adjust_sharpness(tensor, sharpness_factor=2)
    return resize2.transform(resize1.transform(sharpened, {}), {})


def display_reverse(images: List, to_file=None):
    fig, axes = plt.subplots(1, len(images), figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i]

        while x.dim() > 2:
            x = x.squeeze(0)

        x = x.numpy()
        ax.imshow(x, vmin=0, vmax=1)
        ax.axis("off")

    if to_file is not None:
        path = f"{to_file}.png"
        plt.savefig(path, dpi=350)
        plt.close()
        print("Rendered PNG", path)
    else:
        plt.show()


def compute_manhattan_distance(src_mask):
    """
    Manhattan distance from each pixel to the nearest valid pixel (src_mask=True).

    Uses multi-source BFS over 4-connected grid. Correctly handles paths
    that turn corners, unlike the previous row/column decomposition.
    """
    mask = src_mask.cpu().numpy()
    h, w = mask.shape

    INF = h * w + 1
    dist = np.full((h, w), INF, dtype=np.int32)
    q = deque()

    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                dist[y, x] = 0
                q.append((y, x))

    while q:
        cy, cx = q.popleft()
        nd = dist[cy, cx] + 1
        if cy > 0 and dist[cy - 1, cx] > nd:
            dist[cy - 1, cx] = nd
            q.append((cy - 1, cx))
        if cy + 1 < h and dist[cy + 1, cx] > nd:
            dist[cy + 1, cx] = nd
            q.append((cy + 1, cx))
        if cx > 0 and dist[cy, cx - 1] > nd:
            dist[cy, cx - 1] = nd
            q.append((cy, cx - 1))
        if cx + 1 < w and dist[cy, cx + 1] > nd:
            dist[cy, cx + 1] = nd
            q.append((cy, cx + 1))

    return tensor(dist)


def compute_threshold(src_mask):
    print(src_mask.shape)
    dim_y = src_mask.shape[-2]
    dim_x = src_mask.shape[-1]
    print("dim x", dim_x, "dim_y", dim_y)

    return ceil(max(dim_x * 0.5, dim_y * 0.5))


# We don't want to infill everything - some regions like large bodies of water
# do no infer well and so we fill them in heuristically rather than using the model.
#
# To decide which regions to keep after infilling, we use this mask
def compute_edge_reachable(src_mask):
    """
    NaN pixels that are 4-connected to the image border via NaN-only paths.

    Regions touching the border (skies, seas) should be flood-filled, not
    inferred, so they are excluded from the infill keep mask.
    """
    mask = src_mask.cpu().numpy()
    h, w = mask.shape
    reachable = np.zeros((h, w), dtype=bool)
    q = deque()

    for y in range(h):
        for x in (0, w - 1):
            if not mask[y, x] and not reachable[y, x]:
                reachable[y, x] = True
                q.append((y, x))

    for x in range(1, w - 1):
        for y in (0, h - 1):
            if not mask[y, x] and not reachable[y, x]:
                reachable[y, x] = True
                q.append((y, x))

    while q:
        cy, cx = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if (
                0 <= ny < h
                and 0 <= nx < w
                and not mask[ny, nx]
                and not reachable[ny, nx]
            ):
                reachable[ny, nx] = True
                q.append((ny, nx))

    return tensor(reachable)


def compute_infill_keep_mask(src_mask):
    threshold = compute_threshold(src_mask)
    keep = logical_and(
        logical_not(src_mask), compute_manhattan_distance(src_mask) <= threshold
    )
    keep = logical_and(keep, logical_not(compute_edge_reachable(src_mask)))
    return keep


# To improve output we make sure we overlap some of the rows and columns with previous inferences during the tiled inference
min_tile_overlap = 64


def tiled_inference(src_frame, src_mask, src_keep, model, kernel_size, device):
    src_frame = src_frame.to(device)
    src_mask = src_mask.to(device)
    src_keep = src_keep.to(device)

    src_height = src_frame.shape[-2]
    src_width = src_frame.shape[-1]

    print("Beginning tiled inference", src_frame.shape, src_mask.shape, src_keep.shape)

    for start_y in range(0, src_frame.shape[-2], kernel_size - min_tile_overlap):
        print("Start y", start_y)

        for start_x in range(0, src_frame.shape[-1], kernel_size - min_tile_overlap):
            print("Doing", start_x, start_y)

            last_x = False
            last_y = False

            if start_x + kernel_size >= src_width:
                start_x -= start_x + kernel_size - src_width
                last_x = True

            if start_y + kernel_size >= src_height:
                start_y -= start_y + kernel_size - src_height
                last_y = True

            raise Exception(start_x, start_y)

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
                sum(tiles_that_dont_need_inference) != tile_mask.flatten().shape[0]
            )

            if need_to_do_inference:
                print("Inferring tile", start_x, start_y, tile_data.shape)
                print(tile_data.shape, tile_mask.shape)

                inference = model.infer(tile_data, tile_mask, device=device)
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
                    :write_back_height,
                    :write_back_width,
                ]

                src_frame[
                    :,
                    0:1,
                    write_back_start_y:write_back_end_y,
                    write_back_start_x:write_back_end_x,
                ] = reconstructed_frame[:, 0:1, :, :]

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

                display_reverse(
                    [
                        tile_data.to("cpu")[:, 0:1, :, :],
                        tile_mask.to("cpu"),
                        inference.to("cpu")[:, 0:1, :, :],
                        tile_keep.to("cpu"),
                        reconstructed_frame.to("cpu")[:, 0:1, :, :],
                        src_frame.to("cpu")[:, 0:1, :, :],
                        src_mask.to("cpu"),
                    ],
                    to_file=f"{start_y}_{start_x}",
                )
            else:
                print("Skipping tile", start_x, start_y)

    return src_frame


def centered_tiled_inference(src_frame, src_mask, src_keep, model, kernel_size, device):
    src_frame = src_frame.to(device)
    src_mask = src_mask.to(device)
    src_keep = src_keep.to(device)

    src_height = src_frame.shape[-2]
    src_width = src_frame.shape[-1]
    stride = kernel_size - min_tile_overlap

    print(
        "Beginning centered tiled inference",
        src_frame.shape,
        src_mask.shape,
        src_keep.shape,
    )

    # Enumerate all unique tile positions
    raw_positions = set()
    for y in range(0, src_height, stride):
        for x in range(0, src_width, stride):
            start_x = x
            start_y = y
            if start_x + kernel_size > src_width:
                start_x = src_width - kernel_size
            if start_y + kernel_size > src_height:
                start_y = src_height - kernel_size
            raw_positions.add((start_x, start_y))
    tile_positions = sorted(raw_positions)
    print(f"Total unique tiles: {len(tile_positions)}")

    # Build adjacency: neighbors at stride offset
    pos_to_idx = {pos: i for i, pos in enumerate(tile_positions)}
    neighbors = [[] for _ in range(len(tile_positions))]
    for i, (sx, sy) in enumerate(tile_positions):
        for dx, dy in [(-stride, 0), (stride, 0), (0, -stride), (0, stride)]:
            nx, ny = sx + dx, sy + dy
            if (nx, ny) in pos_to_idx:
                neighbors[i].append(pos_to_idx[(nx, ny)])

    # Find seed tile: the one with the most valid (non-masked) pixels
    best_score = -1
    best_idx = 0
    for i, (sx, sy) in enumerate(tile_positions):
        tile_mask = src_mask[:, :, sy : sy + kernel_size, sx : sx + kernel_size]
        valid_count = tile_mask.sum().item()
        if valid_count > best_score:
            best_score = valid_count
            best_idx = i

    print(f"Seed tile: {tile_positions[best_idx]} with {best_score} valid pixels")

    # Priority-queue expansion: highest valid-count first
    pq = []
    processed = set()
    in_queue = set()

    def try_enqueue(idx):
        if idx in processed or idx in in_queue:
            return
        sx, sy = tile_positions[idx]
        tile_mask = src_mask[:, :, sy : sy + kernel_size, sx : sx + kernel_size]
        valid_count = int(tile_mask.sum().item())
        heapq.heappush(pq, (-valid_count, idx))
        in_queue.add(idx)

    try_enqueue(best_idx)
    seq = 0

    while pq:
        _neg_priority, idx = heapq.heappop(pq)
        if idx in processed:
            continue

        start_x, start_y = tile_positions[idx]
        end_x = start_x + kernel_size
        end_y = start_y + kernel_size

        tile_data = src_frame[:, :, start_y:end_y, start_x:end_x]
        tile_mask = src_mask[:, :, start_y:end_y, start_x:end_x]
        tile_keep = src_keep[:, :, start_y:end_y, start_x:end_x]

        print(f"Processing tile [{seq}] ({start_x}, {start_y})")

        # Only run inference if there are keep-pixels still masked
        if logical_and(tile_keep, logical_not(tile_mask)).any():
            print(f"Inferring tile ({start_x}, {start_y})", tile_data.shape)

            inference = model.infer(tile_data, tile_mask, device=device)

            reconstructed_frame = (tile_data * logical_not(tile_keep)) + (
                inference * tile_keep
            )

            # Write back full tile (height channel only, preserve classification)
            src_frame[:, 0:1, start_y:end_y, start_x:end_x] = reconstructed_frame[
                :, 0:1, :, :
            ]
            src_mask[:, :, start_y:end_y, start_x:end_x] = ones(
                (1, 1, kernel_size, kernel_size), dtype=torch.bool, device=device
            )
            src_keep[:, :, start_y:end_y, start_x:end_x] = zeros(
                (1, 1, kernel_size, kernel_size), dtype=torch.bool, device=device
            )

            display_reverse(
                [
                    tile_data.to("cpu")[:, 0:1, :, :],
                    tile_mask.to("cpu"),
                    inference.to("cpu")[:, 0:1, :, :],
                    tile_keep.to("cpu"),
                    reconstructed_frame.to("cpu")[:, 0:1, :, :],
                    src_frame.to("cpu")[:, 0:1, :, :],
                    src_mask.to("cpu"),
                ],
                to_file=f"centered_{seq:04d}",
            )
        else:
            print(
                f"Skipping tile ({start_x}, {start_y}) — all keep pixels already valid"
            )

        processed.add(idx)
        seq += 1

        for ni in neighbors[idx]:
            try_enqueue(ni)

    return src_frame[:, 0:1, :, :]


def whole_datasource_tiled_inference(
    src_frame,
    src_mask,
    kernel_size,
    model,
    device=None,
    num_time_steps: int = num_time_steps,
    ema_decay: float = 0.9999,
):
    print("Starting to compute infill mask", src_mask.shape)
    keep_mask = compute_infill_keep_mask(src_mask[0][0]).reshape(src_mask.shape)
    print("Computed infill mask")

    result = tiled_inference(
        src_frame.clone(), src_mask, keep_mask, model, kernel_size, device
    )
    print("Inf result", result.shape)

    # result = aa(result)
    # print("AA result", result.shape)

    display_reverse(
        [
            src_frame.to("cpu")[:, 0:1, :, :],
            src_mask.to("cpu"),
            keep_mask.to("cpu"),
            result.to("cpu")[:, 0:1, :, :],
        ],
        to_file="./out",
    )

    return result


def simple_grid_tiled_inference(
    src_frame, src_mask, src_keep, model, kernel_size, device
):
    src_frame = src_frame.to(device)
    src_mask = src_mask.to(device)
    src_keep = src_keep.to(device)

    src_height = src_frame.shape[-2]
    src_width = src_frame.shape[-1]
    stride = kernel_size

    print(
        "Beginning simple grid tiled inference",
        src_frame.shape,
        src_mask.shape,
        src_keep.shape,
    )

    display_reverse(
        [
            src_frame.to("cpu")[:, 0:1, :, :],
            src_frame.to("cpu")[:, 1:2, :, :],
        ],
        to_file="start",
    )

    for start_y in range(0, src_height, stride):
        for start_x in range(0, src_width, stride):
            end_x = start_x + kernel_size
            end_y = start_y + kernel_size

            if end_x > src_width:
                start_x = src_width - kernel_size
                end_x = src_width
            if end_y > src_height:
                start_y = src_height - kernel_size
                end_y = src_height

            tile_data = src_frame[:, :, start_y:end_y, start_x:end_x]
            tile_mask = src_mask[:, :, start_y:end_y, start_x:end_x]
            tile_keep = src_keep[:, :, start_y:end_y, start_x:end_x]

            print(f"Tile ({start_x}, {start_y})")

            class_ch = tile_data[:, 1:2, :, :]
            print(
                f"  classification channel: min={class_ch.min().item():.3f} max={class_ch.max().item():.3f} mean={class_ch.mean().item():.3f} unique={torch.unique(class_ch).tolist()}"
            )

            if logical_and(tile_keep, logical_not(tile_mask)).any():
                print(f"Inferring tile ({start_x}, {start_y})", tile_data.shape)

                inference = model.infer(tile_data, tile_mask, device=device)

                print("Recombining inference", tile_data.shape, inference.shape)

                origin_frame = tile_data[:, 0:1, :, :]

                reconstructed_frame = (origin_frame * logical_not(tile_keep)) + (
                    inference * tile_keep
                )

                print(
                    f"  inference: min={inference.min().item():.4f} max={inference.max().item():.4f} mean={inference.mean().item():.4f}"
                )
                print(
                    f"  reconstructed: min={reconstructed_frame.min().item():.4f} max={reconstructed_frame.max().item():.4f}"
                )

                src_frame[:, 0:1, start_y:end_y, start_x:end_x] = reconstructed_frame
                src_mask[:, :, start_y:end_y, start_x:end_x] = ones(
                    (1, 1, kernel_size, kernel_size), dtype=torch.bool, device=device
                )
                src_keep[:, :, start_y:end_y, start_x:end_x] = zeros(
                    (1, 1, kernel_size, kernel_size), dtype=torch.bool, device=device
                )

                display_reverse(
                    [
                        tile_data.to("cpu")[:, 0:1, :, :],
                        tile_data.to("cpu")[:, 1:2, :, :],
                        tile_mask.to("cpu"),
                        inference.to("cpu")[:, 0:1, :, :],
                        tile_keep.to("cpu"),
                        reconstructed_frame.to("cpu")[:, 0:1, :, :],
                        src_frame.to("cpu")[:, 0:1, :, :],
                        src_mask.to("cpu"),
                    ],
                    to_file=f"grid_{start_y}_{start_x}",
                )

            else:
                print(f"Skipping tile ({start_x}, {start_y})")

    return src_frame


def whole_datasource_simple_grid_inference(
    src_frame,
    src_mask,
    kernel_size,
    model,
    device=None,
    num_time_steps: int = num_time_steps,
    ema_decay: float = 0.9999,
    denoise_sigma: float = 0.5,
    median_kernel: int = 4,
):
    print("Starting to compute infill mask", src_mask.shape)

    keep_mask = compute_infill_keep_mask(src_mask[0][0]).reshape(src_mask.shape)
    pixels_that_will_be_missing_after_inference = logical_and(
        logical_not(src_mask), logical_not(keep_mask)
    )
    print("Computed infill mask")

    result_before_flood = simple_grid_tiled_inference(
        src_frame.clone(), src_mask, keep_mask, model, kernel_size, device
    )
    print("inference result", result_before_flood.shape)

    result = flood_fill_nans(
        result_before_flood.clone(), pixels_that_will_be_missing_after_inference
    )
    print("Post-flood-fill result", result.shape)

    if median_kernel > 1:
        pad = median_kernel // 2
        height_ch = result[:, 0:1, :, :]
        padded = torch.nn.functional.pad(
            height_ch, (pad, pad, pad, pad), mode="replicate"
        )
        patches = padded.unfold(2, median_kernel, 1).unfold(3, median_kernel, 1)
        result[:, 0:1, :, :] = (
            patches.contiguous()
            .view(*height_ch.shape[:2], height_ch.shape[2], height_ch.shape[3], -1)
            .median(dim=-1)
            .values
        )
        print(f"Median filtered with kernel={median_kernel}")

    if denoise_sigma > 0:
        kernel_size_denoise = max(3, int(denoise_sigma * 3) * 2 + 1)
        blur = GaussianBlur(kernel_size=kernel_size_denoise, sigma=denoise_sigma)
        result[:, 0:1, :, :] = blur(result[:, 0:1, :, :])
        print(f"Denoised with sigma={denoise_sigma}, kernel={kernel_size_denoise}")

    display_reverse(
        [
            src_frame.to("cpu")[:, 0:1, :, :],
            src_mask.to("cpu"),
            keep_mask.to("cpu"),
            result_before_flood.to("cpu")[:, 0:1, :, :],
            result.to("cpu")[:, 0:1, :, :],
        ],
        to_file="./out_grid",
    )

    return result


def _label_connected_components(mask):
    """Label 4-connected components in a 2D boolean mask using BFS."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] and labels[y, x] == 0:
                current_label += 1
                queue = [(y, x)]
                labels[y, x] = current_label
                idx = 0
                while idx < len(queue):
                    cy, cx = queue[idx]
                    idx += 1
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and mask[ny, nx]
                            and labels[ny, nx] == 0
                        ):
                            labels[ny, nx] = current_label
                            queue.append((ny, nx))
    return labels, current_label


def _binary_dilate(mask):
    """Dilate a binary mask by 1 pixel using 4-connectivity (shifted views)."""
    result = mask.copy()
    result[1:] |= mask[:-1]
    result[:-1] |= mask[1:]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    return result


def flood_fill_nans(frame, missing):
    device = frame.device
    dtype = frame.dtype

    if not missing.any():
        return frame

    height = frame[:, 0:1, :, :]
    h = height[0, 0].cpu().numpy()
    m = missing[0, 0].cpu().numpy()
    result = h.copy()

    valid_mask = ~m
    range_val = (
        result[valid_mask].max() - result[valid_mask].min() if valid_mask.any() else 0.0
    )
    fill_offset = range_val * 0.01

    labels, num_features = _label_connected_components(m)

    for i in range(1, num_features + 1):
        component = labels == i

        dilated = _binary_dilate(component)
        boundary = dilated & ~component & ~m

        if boundary.any():
            avg = np.mean(result[boundary])
            result[component] = avg - fill_offset
        else:
            valid = ~m
            if valid.any():
                result[component] = np.mean(result[valid]) - fill_offset
            else:
                result[component] = 0.0

    frame[:, 0:1, :, :] = torch.tensor(result, device=device, dtype=dtype).view(
        1, 1, *result.shape
    )

    return frame


def whole_datasource_centered_tiled_inference(
    src_frame,
    src_mask,
    kernel_size,
    model,
    device=None,
    num_time_steps: int = num_time_steps,
    ema_decay: float = 0.9999,
):
    print("Starting to compute infill mask", src_mask.shape)

    keep_mask = compute_infill_keep_mask(src_mask[0][0]).reshape(src_mask.shape)
    pixels_that_will_be_missing_after_inference = logical_and(
        logical_not(src_mask), logical_not(keep_mask)
    )
    print("Computed infill mask")

    result_before_flood = centered_tiled_inference(
        src_frame.clone(), src_mask, keep_mask, model, kernel_size, device
    )
    print("inference result", result_before_flood.shape)

    result = flood_fill_nans(
        result_before_flood.clone(), pixels_that_will_be_missing_after_inference
    )
    print("Post-flood-fill result", result.shape)

    display_reverse(
        [
            src_frame.to("cpu")[:, 0:1, :, :],
            src_mask.to("cpu"),
            keep_mask.to("cpu"),
            result_before_flood.to("cpu")[:, 0:1, :, :],
            result.to("cpu")[:, 0:1, :, :],
        ],
        to_file="./out_centered",
    )

    return result
