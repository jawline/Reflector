import random
import numpy as np
import torch
from torch import (
    logical_and,
    randperm,
)
from torch.distributions import Normal
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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


def kl_loss(encoded_distribution, sample, *, beta=None):
    q_dist = Normal(
        encoded_distribution.mean, torch.exp(0.5 * encoded_distribution.log_var)
    )
    p_dist = Normal(
        torch.zeros_like(encoded_distribution.mean),
        torch.ones_like(encoded_distribution.log_var),
    )

    sample = encoded_distribution.sample()
    log_qdist = q_dist.log_prob(sample)
    log_pdist = p_dist.log_prob(sample)
    kl_div = log_qdist - log_pdist

    # print(kl_div.shape)

    kl_div = kl_div.sum(-1).mean()
    # print(kl_div.item())

    return kl_div * beta


# Mixes the noise from other entries in the batch so we get a realistic but lossable missing data
# The mask of each entry in a batch is mixed with the mask from a different entry in the same batch
def apply_batch_noise(masks, count):
    for i in range(0, count):
        idx = randperm(masks.shape[0], device=masks.device)
        shuffled = masks.index_select(0, idx)
        masks = logical_and(masks, shuffled)

    return masks


def display_images(images, to_file=None):
    fig, axes = plt.subplots(1, len(images), figsize=(len(images), 1))
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
