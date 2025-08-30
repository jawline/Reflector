import pickle
from torch import tensor, logical_not, masked_select, cat, isnan, nan_to_num
from torch.utils.data import Dataset
from glob import glob
from math import floor, ceil
from random import Random
from tqdm import tqdm
from constants import tile_width, tile_height
from functools import lru_cache
from math import nan

max_attempts = 50


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
@lru_cache(maxsize=16)
def load(path):
    with open(path, "rb") as file:
        sample = pickle.load(file)

    width = sample["width"]
    height = sample["height"]

    with_nan = tensor([x if x != None else nan for x in sample["data"]]).reshape(
        (height, width)
    )

    nans = isnan(with_nan)
    mask = logical_not(nans)

    without_nan = nan_to_num(with_nan, nan=0.0)

    sample["without_nan"] = without_nan
    sample["mask"] = mask

    return sample


# Dataset for candidate selection, used to preprocess actual samples out of larger files by preprocess_samples
# Expensive, so best to preprocess and then used pre serialized samples
class TerrainDatasetSlow(Dataset):
    def __init__(self, samples_dir):
        self.sample_x = tile_width
        self.sample_y = tile_height
        files = glob(f"{samples_dir}/**/**.pre.pt", recursive=True)
        final_files = []
        print("Loading", len(files), " files")
        num_files = len(files)
        for i, path in tqdm(enumerate(files), desc=f"Num Files {num_files}"):

            try:
                with open(path, "rb") as file:
                    data = pickle.load(file)

                width = data["width"]
                height = data["height"]

                samples = ceil((width / self.sample_x) * (height / self.sample_y))

                if width > self.sample_x and height > self.sample_y:
                    # Insert the same file in samples times to generate samples random samples
                    for _i in range(samples):
                        final_files.append(path)
            except Exception as e:
                print("path failed to load", path)

        self.files = final_files
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

        without_nan = sample["without_nan"][start_y:end_y, start_x:end_x]
        mask = sample["mask"][start_y:end_y, start_x:end_x]

        return without_nan, mask

    def reject_candidate(self, mask):
        mask = mask.flatten()
        num_nans = sum(logical_not(mask))
        ratio_of_nans = num_nans / len(mask)
        return ratio_of_nans > 0.2

    def __getitem__(self, idx):
        # Make this deterministic for a given self.files by seeding rand per iteration with the idx
        rand = Random(idx)
        sample = load(self.files[idx])
        without_nan = tensor([])
        mask = tensor([])
        attempts = 0

        # To avoid raising, instead expect the caller to check broken before serializing
        broken = self.files[idx] in self.broken

        while not broken:
            without_nan, mask = self.candidate(rand, sample)
            if not self.reject_candidate(mask):
                break
            attempts += 1
            if attempts > max_attempts:
                broken = True
                self.broken.add(self.files[idx])

        return {
            "without_nan": without_nan,
            "mask": mask,
            "broken": broken,
        }
