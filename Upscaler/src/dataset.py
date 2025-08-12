from torch import tensor, logical_not, load, masked_select, cat
from torch.utils.data import Dataset
from glob import glob
from sample_loader import load_sample, tell_width_height, prepare_tensor
from math import floor, ceil, isnan
from random import Random
from tqdm import tqdm
from constants import tile_width, tile_height

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


# Dataset for candidate selection, used to preprocess actual samples out of larger files by preprocess_samples
# Expensive, so best to preprocess and then used pre serialized samples
class TerrainDatasetSlow(Dataset):
    def __init__(self, samples_dir):
        self.sample_x = tile_width
        self.sample_y = tile_height
        files = glob(f"{samples_dir}/**/**.h3t", recursive=True)
        final_files = []
        print("Loading", len(files), " files")
        num_files = len(files)
        for i, file in tqdm(enumerate(files), desc=f"Num Files {num_files}"):
            width, height = tell_width_height(file)
            samples = ceil((width / self.sample_x) * (height / self.sample_y))
            if width > self.sample_x and height > self.sample_y:
                # Insert the same file in samples times to generate samples random samples
                for _i in range(samples):
                    final_files.append(file)
        self.files = final_files

    def __len__(self):
        return len(self.files)

    def candidate(self, rand, sample):
        start_x = rand.randint(0, sample.width - self.sample_x)
        start_y = rand.randint(0, sample.height - self.sample_y)
        with_nan = sample.tensor(start_x, start_y, self.sample_x, self.sample_y)
        without_nan, mask = prepare_tensor(with_nan)
        return with_nan, without_nan, mask

    def reject_candidate(self, mask):
        num_nans = 0
        elts = 0
        for row in mask:
            for val in row:
                num_nans += int(not val)
                elts += 1
        ratio_of_nans = num_nans / elts
        return ratio_of_nans > 0.4

    def __getitem__(self, idx):
        # Make this deterministic for a given self.files by seeding rand per iteration with the idx
        rand = Random(idx)
        sample = load_sample(self.files[idx])
        with_nan = None
        without_nan = None
        mask = None
        attempts = 0

        # To avoid raising, instead expect the caller to check broken before serializing
        broken = False
        while not broken:
            with_nan, without_nan, mask = self.candidate(rand, sample)
            if not self.reject_candidate(mask):
                break
            attempts += 1
            if attempts > max_attempts:
                broken = True

        # Pre-normalize the inputs
        without_nan = without_nan.reshape((1, 1, self.sample_x, self.sample_y))
        mask = mask.reshape(without_nan.shape)
        without_nan, max_values, min_values = norm(without_nan, mask)
        without_nan = without_nan.reshape((self.sample_x, self.sample_y))
        mask = mask.reshape(without_nan.shape)

        return {
            "mask": mask,
            "without_nan": without_nan,
            "min_values": min_values,
            "max_values": max_values,
            "broken": broken,
        }


class TerrainDataset(Dataset):
    def __init__(self, samples_dir):
        self.files = glob(f"{samples_dir}/*.pt")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        result = load(self.files[idx])
        return result
