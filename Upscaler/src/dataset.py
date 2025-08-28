from torch import load
from torch.utils.data import Dataset
from glob import glob


class TerrainDataset(Dataset):
    def __init__(self, samples_dir):
        self.files = glob(f"{samples_dir}/*.pt")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        result = load(self.files[idx])
        return result
