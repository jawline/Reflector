import sys
import torch
from train import train
from dataset import TerrainDataset
from infer import generative_inference


def main():
    print("Torch version", torch.__version__)

    mode = sys.argv[-2]
    dataset = sys.argv[-1]

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        print("Apple Silicon acceleration possible")
        device = torch.device("mps")

    if mode == "train":
        print("Loading", dataset)
        dataset = TerrainDataset(dataset)
        train(dataset, checkpoint_path=None, lr=2e-5, num_epochs=75, device=device)
    elif mode == "infer":
        generative_inference(dataset)


if __name__ == "__main__":
    main()
