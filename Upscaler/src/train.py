import random
from torch import (
    logical_and,
)
from tqdm import tqdm

from torch import rand, rand_like, logical_not, transpose, ones_like, zeros
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from infer import display_reverse
from models.util import set_seed, dataloader, apply_batch_noise, display_images


def train(
    model,
    dataset,
    batch_size: int = 8,
    num_epochs: int = 150,
    seed: int = -1,
    checkpoint_path: str = None,
    device=None,
):
    set_seed(random.randint(0, 2**32 - 1)) if seed == -1 else set_seed(seed)
    dataset_len = len(dataset)
    dataset_per_epoch = dataset_len / batch_size
    train_loader = dataloader(dataset, batch_size)

    scheduler = ReduceLROnPlateau(model.optimizer, mode="min", patience=0, factor=0.1)

    for i in range(num_epochs):
        # print("Optimizer state")
        # for param_group in model.optimizer.param_groups:
        #    print(param_group["lr"])

        total_loss = zeros(1, device=device)
        for bidx, datapoint in enumerate(
            tqdm(train_loader, desc=f"Epoch {i + 1}/{num_epochs}")
        ):
            for_train = datapoint["terrain"].to(device, non_blocking=True).squeeze(1)
            mask = datapoint["mask"].to(device, non_blocking=True).squeeze(1)

            #print(for_train.shape, mask.shape)

            if random.choice([True, False]):
                for_train = transpose(for_train, -1, -2)
                mask = transpose(mask, -1, -2)

            batch_noise = apply_batch_noise(mask, count=random.randint(1, 2))

            if random.choice([True, False, False, False, False]):
                min_noise = 0.01
                max_noise = 0.2
                noise_thresh_per_batch_elt = (
                    min_noise + (rand((batch_size, 1, 1, 1)) * (max_noise - min_noise))
                ).to(device)
                additional_noise = (
                    rand_like(batch_noise.float()) > noise_thresh_per_batch_elt
                )
            else:
                additional_noise = ones_like(batch_noise)

            total_mask = logical_and(batch_noise, additional_noise)
            for_autoencoder = for_train * total_mask

            loss = model.train_step(for_autoencoder, total_mask, for_train, mask)
            total_loss += loss.detach()

            if (bidx % int(dataset_per_epoch // 50)) == 0:
                print(
                    "Sample",
                    bidx,
                    loss.item(),
                    for_train.shape[0],
                )

                model.checkpoint(checkpoint_path)

                elt = 0

                test_data = for_autoencoder[elt : elt + 1].to(device)
                test_mask = total_mask[elt : elt + 1].to(device)
                test = model.infer(test_data, test_mask).to("cpu").detach()

                display_images(
                    [
                        test_data.to("cpu")[0][0].detach(),
                        test_mask.to("cpu")[0][0].detach(),
                        test,
                    ],
                    to_file=elt,
                )

        avg_loss = (total_loss / dataset_per_epoch).item()
        scheduler.step(avg_loss)

        print(
            f"Epoch {i + 1} | Loss {(total_loss / (dataset_len / batch_size)).item()} (Saved)"
        )

        if total_loss.item() < 0:
            raise Exception("Explosion - self terminating")

        model.checkpoint(checkpoint_path)
