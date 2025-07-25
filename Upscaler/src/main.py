import sys
import torch
from train import train
from dataset import TerrainDataset
from infer import masked_inference, generative_inference


def main():
    print("Torch version", torch.__version__)

    mode = sys.argv[-3]
    dataset = sys.argv[-2]
    checkpoint = sys.argv[-1]

    device = torch.device("cpu")

    if torch.backends.mps.is_available():
        print("Apple Silicon acceleration possible")
        device = torch.device("mps")

    if torch.cuda.is_available():
        print("CUDA acceleration is possible")
        device = torch.device("cuda")

    print("Loading", dataset)
    dataset = TerrainDataset(dataset)

    if mode == "train":
        print("Starting training")
        train(
            dataset, checkpoint_path=checkpoint, lr=2e-5, num_epochs=75, device=device
        )
    elif mode == "infer":
        print("Starting inference")
        masked_inference(dataset, device=device, checkpoint_path=checkpoint)
    elif mode == "generate":
        print("Starting generative inference")
        generative_inference(device=device, checkpoint_path=checkpoint)


if __name__ == "__main__":
    main()
