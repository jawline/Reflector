from torch import tensor, logical_not, load
from torch.utils.data import Dataset
from glob import glob
from sample_loader import load_sample
from math import floor, isnan
from random import Random


# Dataset for candidate selection, used to preprocess actual samples out of larger files by preprocess_samples
# Expensive, so best to preprocess and then used pre serialized samples
class TerrainDatasetSlow(Dataset):
    def __init__(self, samples_dir, samples_per_item=256, sample_x=64, sample_y=64):
        self.samples_per_item = samples_per_item
        self.sample_x = sample_x
        self.sample_y = sample_y
        self.files = glob(f"{samples_dir}/*.h3t")

    def __len__(self):
        return len(self.files) * self.samples_per_item

    def candidate(self, rand, sample):
        start_x = rand.randint(0, sample.width - self.sample_x)
        start_y = rand.randint(0, sample.height - self.sample_y)
        with_nan = sample.tensor(start_x, start_y, self.sample_x, self.sample_y)
        mask = tensor([[isnan(x) for x in row] for row in with_nan[0]]).reshape(
            with_nan.shape
        )
        mask = logical_not(mask)
        without_nan = sample.tensor(
            start_x, start_y, self.sample_x, self.sample_y, replace_nan_with=0.0
        )
        return with_nan, without_nan, mask

    def reject_candidate(self, mask):
        num_nans = 0
        elts = 0
        for _, row in enumerate(mask[0]):
            for _, val in enumerate(row):
                num_nans += int(not val)
                elts += 1
        ratio_of_nans = num_nans / elts
        return ratio_of_nans > 0.2

    def __getitem__(self, idx):
        rand = Random(idx)
        idx = floor(idx / self.samples_per_item)
        sample = load_sample(self.files[idx])
        with_nan = None
        without_nan = None
        mask = None
        while True:
            with_nan, without_nan, mask = self.candidate(rand, sample)
            if not self.reject_candidate(mask):
                break

        return {
            "mask": mask,
            "without_nan": without_nan,
        }


class TerrainDataset(Dataset):
    def __init__(self, samples_dir, samples_per_item=256, sample_x=64, sample_y=64):
        self.files = glob(f"{samples_dir}/*.pt")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return load(self.files[idx])
