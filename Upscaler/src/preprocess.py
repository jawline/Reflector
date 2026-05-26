import pickle
import functools
from torch import tensor, logical_not, masked_select, isnan, nan_to_num, flip
from torch.utils.data import Dataset
from glob import glob
from random import Random
from constants import tile_size
from math import nan

samples_per_file = 16


def norm(inp, mask):
    mins = []
    maxs = []
    for image, mask in zip(inp, mask):
        select = masked_select(image, mask)
        if len(select) == 0:
            select = image.flatten()
        min_value, max_value = select.aminmax(dim=-1, keepdim=True)
        mins.append(min_value)
        maxs.append(max_value)

    mins = tensor(mins).to(inp.device).reshape((inp.shape[0], 1, 1, 1))
    maxs = tensor(maxs).to(inp.device).reshape((inp.shape[0], 1, 1, 1))
    delta = maxs - mins
    output = (inp - mins) / delta
    return output, mins, maxs


def unnorm(image, min_values, max_values):
    delta = max_values - min_values
    image = image * delta
    return image + min_values


# LRU cache the result so indexing into the same file is cheaper
@functools.lru_cache(maxsize=8)
def load(path):
    with open(path, "rb") as file:
        sample = pickle.load(file)

    heightmap = sample["heightmap"]

    width = heightmap["width"]
    height = heightmap["height"]

    with_nan = tensor([x if x != None else nan for x in heightmap["data"]]).reshape(
        (height, width)
    )
    nans = isnan(with_nan)
    without_nan = nan_to_num(with_nan, nan=0.0)

    mask = logical_not(nans)

    heightmap["mask"] = mask
    heightmap["without_nan"] = without_nan

    return heightmap


class TerrainDatasetSlow(Dataset):
    def __init__(self, samples_dir):
        self.sample_x = tile_size
        self.sample_y = tile_size
        # Sort for deterministic ordering between workers
        files = sorted(glob(f"{samples_dir}/**/**.pre.pt", recursive=True))
        self.files = [f for f in files for _ in range(samples_per_file)]
        self.broken = set()

    def __len__(self):
        return len(self.files)

    def candidate(self, rand, sample):
        width = sample["width"]
        height = sample["height"]

        start_x = rand.randint(0, width - self.sample_x)
        start_y = rand.randint(0, height - self.sample_y)

        end_x = start_x + self.sample_x
        end_y = start_y + self.sample_y

        without_nan = sample["without_nan"][start_y:end_y, start_x:end_x].clone()
        mask = sample["mask"][start_y:end_y, start_x:end_x].clone()

        terrain = without_nan.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return terrain, mask

    def reject_candidate(self, mask):
        mask = mask.flatten()
        num_nans = sum(logical_not(mask))
        ratio_of_nans = num_nans / len(mask)
        return ratio_of_nans > 0.2

    def __getitem__(self, idx):
        rand = Random(idx)
        terrain = tensor([])
        mask = tensor([])

        if self.files[idx] in self.broken:
            return {"terrain": terrain, "mask": mask, "broken": True}

        sample = load(self.files[idx])

        if sample["width"] < self.sample_x or sample["height"] < self.sample_y:
            self.broken.add(self.files[idx])
            return {"terrain": terrain, "mask": mask, "broken": True}

        terrain, mask = self.candidate(rand, sample)

        if self.reject_candidate(mask):
            self.broken.add(self.files[idx])
            return {"terrain": tensor([]), "mask": tensor([]), "broken": True}

        return {"terrain": terrain, "mask": mask, "broken": False}
