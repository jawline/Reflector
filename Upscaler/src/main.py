import sys
import torch
from train import train
from dataset import TerrainDataset
from infer import masked_inference, generative_inference
from sample_loader import load_sample, prepare_tensor
from constants import tile_width, tile_height


def select_device():
    device = torch.device("cpu")

    if torch.backends.mps.is_available():
        print("Apple Silicon acceleration possible")
        device = torch.device("mps")

    if torch.cuda.is_available():
        print("CUDA acceleration is possible")
        device = torch.device("cuda")

    return device


def main():
    print("Torch version", torch.__version__)

    mode = sys.argv[-3]
    dataset = sys.argv[-2]
    checkpoint = sys.argv[-1]

    device = select_device()

    if mode == "train":
        print("Starting training")
        print("Loading", dataset)
        dataset = TerrainDataset(dataset)
        train(
            dataset, checkpoint_path=checkpoint, lr=2e-5, num_epochs=75, device=device
        )
    elif mode == "infer":
        print("Starting inference")
        print("Loading", dataset)
        dataset = TerrainDataset(dataset)
        masked_inference(dataset, device=device, checkpoint_path=checkpoint)
    elif mode == "generate":
        print("Starting generative inference")
        generative_inference(device=device, checkpoint_path=checkpoint)
    elif mode == "whole-frame":
        path = dataset
        sample = load_sample(path)
        src_frame, mask = prepare_tensor(
            sample.tensor(0, 0, sample.width, sample.height)
        )
        inferred_frame = whole_datasource_tiled_inference(
            src_frame,
            mask,
            tile_width,
            tile_height,
            device=device,
            checkpoint_path=checkpoint,
        )
        raise Exception("TODO: Serialize finished result")


if __name__ == "__main__":
    main()
