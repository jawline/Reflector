import sys
import torch
from tqdm import tqdm
from dataset import TerrainDatasetSlow


def main():
    print("Torch version", torch.__version__)

    from_ = sys.argv[-2]
    to_ = sys.argv[-1]

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        print("Apple Silicon acceleration possible")
        device = torch.device("mps")

    dataset = TerrainDatasetSlow(from_)

    for i, element in tqdm(enumerate(dataset)):
        torch.save(element, f"{to_}/{i}.pt")


if __name__ == "__main__":
    main()
