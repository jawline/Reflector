import sys
import torch
import pickle
from torch import tensor
from train import train, train_auto
from dataset import TerrainDataset
from infer import (
    masked_inference,
    generative_inference,
    whole_datasource_tiled_inference,
)
from constants import tile_width, tile_height
from infer import display_reverse
from math import nan
from torch import isnan, logical_not, nan_to_num
import preprocess


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
    elif mode == "train-auto":
        dataset = TerrainDataset(dataset)
        train_auto(
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
        print("Preparing")

        sample = preprocess.load(path)
        
        without_nan = sample['without_nan'].unsqueeze(0).unsqueeze(0)
        mask = sample['mask'].unsqueeze(0).unsqueeze(0)

        print(without_nan.shape, mask.shape)

        #display_reverse([without_nan, mask])

        inferred_frame = whole_datasource_tiled_inference(
            without_nan,
            mask,
            tile_width,
            tile_height,
            device=device,
            checkpoint_path=checkpoint,
        )

        result = {
            "data": inferred_frame.flatten().tolist(),
            "width": sample['width'],
            "height": sample['height'],
            "scale_z": sample['scale_z'],
        }

        with open("out.pt", "wb") as f:
            pickle.dump(result, f)


if __name__ == "__main__":
    main()
