import sys
import torch
import pickle
from torch import cat
from train import train
from dataset import TerrainDataset
from infer import (
    masked_inference,
    generative_inference,
)
from tiled_inference import whole_datasource_tiled_inference
from constants import tile_size
from models.diffusion import Model as SimpleModel
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
        model = SimpleModel(lr=1e-4, device=device)
        model.load(checkpoint)
        train(model, dataset, num_epochs=75, device=device, checkpoint_path=checkpoint)
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
        without_nan = sample["without_nan"].unsqueeze(0).unsqueeze(0)
        classification = sample["classification"].unsqueeze(0).unsqueeze(0)
        mask = sample["mask"].unsqueeze(0).unsqueeze(0)

        # Merge the classification channel in
        without_nan = cat([without_nan, classification], dim=1)

        print("Prepared", without_nan.shape, mask.shape)


        model = SimpleModel(lr=1e-4, device=device)
        model.load(checkpoint)

        # display_reverse([without_nan, mask])

        inferred_frame = whole_datasource_tiled_inference(
            without_nan,
            mask,
            tile_size,
            device=device,
            model=model,
        )

        print(inferred_frame.shape)

        result = {
            "data": inferred_frame.flatten().tolist(),
            "width": inferred_frame.shape[-1],
            "height": inferred_frame.shape[-2],
            "scale_z": sample["scale_z"],
        }

        with open("out.pt", "wb") as f:
            pickle.dump(result, f)
            print("Wrote output")


if __name__ == "__main__":
    main()
