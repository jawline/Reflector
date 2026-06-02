import matplotlib.pyplot as plt
from typing import List
from torch import (
    tensor,
    no_grad,
    logical_not,
    rand,
    sum,
)
from constants import num_time_steps


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


def prepare_model(device, checkpoint_path, ema_decay, num_time_steps):
    # scheduler, model, optimizer, ema, scaler = model_loader.load(
    #    device, checkpoint_path, ema_decay, num_time_steps, lr=0.1
    # )
    # return ema, scheduler
    return None


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
    kernel_size=None,
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
                (1, 1, kernel_size, kernel_size)
            )
            src_mask = datapoint["mask"].reshape((1, 1, kernel_size, kernel_size))
            display_reverse([src_frame.to("cpu"), src_frame.to("cpu")])

            # Skip a candidate if it has a lot of data
            if sum(src_mask) > (kernel_size * kernel_size) * 0.96:
                print("Skip candidate, too nice")
                continue

            images = infer_frame(
                device, src_frame, src_mask, model, scheduler, num_time_steps, times
            )

            display_reverse(images)


def generative_inference(
    kernel_size=None,
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
            z = rand(size=(1, 1, kernel_size, kernel_size))
            mask = tensor([False]).repeat(z.shape)
            images = infer_frame(
                device, z, mask, model, scheduler, num_time_steps, times
            )
            display_reverse(images)
