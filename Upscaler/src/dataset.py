from torch import tensor, logical_not, load
from torch.utils.data import Dataset
from glob import glob
from sample_loader import load_sample, tell_width_height, prepare_tensor
from math import floor, ceil, isnan
from random import Random


max_attempts = 200


# Dataset for candidate selection, used to preprocess actual samples out of larger files by preprocess_samples
# Expensive, so best to preprocess and then used pre serialized samples
class TerrainDatasetSlow(Dataset):
    def __init__(self, samples_dir, sample_x=128, sample_y=128):
        self.sample_x = sample_x
        self.sample_y = sample_y
        files = glob(f"{samples_dir}/**/**.h3t", recursive=True)
        final_files = []
        print("Loading", len(files), " files")
        for i, file in enumerate(files):
            width, height = tell_width_height(file)
            samples = ceil((width / sample_x) * (height / sample_y))
            print(f"{i} num samples {samples}")
            if width > sample_x and height > sample_y:
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
        while True:
            with_nan, without_nan, mask = self.candidate(rand, sample)
            if not self.reject_candidate(mask):
                break
            attempts += 1
            if attempts > max_attempts:
                raise Exception(
                    f"Could not find a good candidate in sample {self.files[idx]}"
                )

        return {
            "mask": mask,
            "without_nan": without_nan,
        }


class TerrainDataset(Dataset):
    def __init__(self, samples_dir):
        self.files = glob(f"{samples_dir}/*.pt")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        result = load(self.files[idx])
        return result
