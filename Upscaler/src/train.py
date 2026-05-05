import random
from torch import (
    logical_and,
)
from tqdm import tqdm

from torch import (
    rand,
    rand_like,
    logical_not,
    transpose,
    ones_like
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from infer import display_reverse
from models.util import set_seed, dataloader, apply_batch_noise, display_images




def train(
    model,
    dataset,
    batch_size: int = 4,
    num_epochs: int = 150,
    seed: int = -1,
    ema_decay: float = 0.9999,
    checkpoint_path: str = None,
    device=None,
):
    set_seed(random.randint(0, 2**32 - 1)) if seed == -1 else set_seed(seed)
    dataset_len = len(dataset)
    dataset_per_epoch = dataset_len / batch_size
    train_loader = dataloader(dataset, batch_size)

    scheduler = ReduceLROnPlateau(
        model.optimizer, mode="min", patience=0, factor=0.1, threshold=0.001
    )


    for i in range(num_epochs):

        print("Optimizer state")
        for param_group in model.optimizer.param_groups:
            print(param_group["lr"])

        total_loss = 0
        for bidx, datapoint in enumerate(
            tqdm(train_loader, desc=f"Epoch {i + 1}/{num_epochs}")
        ):
            model.optimizer.zero_grad(set_to_none=True)
            for_train = (
                datapoint["terrain_with_classification"]
                .to(device, non_blocking=True)
                .squeeze(1)
            )
            mask = datapoint["mask"].to(device, non_blocking=True).squeeze(1)

            # print(for_train.shape, mask.shape)

            # Randomly choose to transpose the X,Y (We could during data generation rotate the entire tile before translating it to a heightmap, but that is trickier)
            if random.choice([True, False]):
                for_train = transpose(for_train, -1, -2)
                mask = transpose(mask, -1, -2)

            # print(for_train.shape, mask.shape)

            # Take a random element from the batch and combine its missing data with our own so that we incorporate some real loooking missing data into our own input
            batch_noise = apply_batch_noise(mask, count=random.randint(1, 2))

            # print(for_train.shape, mask.shape)

            # Generate a bunch of random numbers (batch_size,) between 0 and 1 for a known noise addition
            # Add some more noise to the image so the decoder can see some blank cells
            if random.choice([True, False, False, False, False]):
                min_noise = 0.01
                max_noise = 0.2
                noise_thresh_per_batch_elt = (
                    min_noise
                    + (rand((batch_size, 1, 1, 1)) * (max_noise - min_noise))
                ).to(device)
                additional_noise = rand_like(batch_noise) > noise_thresh_per_batch_elt
            else:
                additional_noise = ones_like(batch_noise)

            total_mask = logical_and(batch_noise, additional_noise)
            for_autoencoder = for_train * total_mask

            print(total_mask.shape, batch_noise.shape, additional_noise.shape)
            output, loss = model.train_step(for_autoencoder, total_mask, for_train[:,0:1,:,:])
            total_loss += loss.item()

            reconstructed = (for_train * mask) + (logical_not(mask) * output)

            if (bidx % int(dataset_per_epoch // 100)) == 0:
                print(
                    "Sample",
                    bidx,
                    loss.item(),
                    for_train.shape[0],
                )

                model.checkpoint(checkpoint_path)

                for elt in range(0, for_train.shape[0]):
                    dec_for_train = for_train.to("cpu")[elt][0].detach()
                    dec_clas_train = for_train.to("cpu")[elt][1].detach()
                    dec_for_autoencoder = for_autoencoder.to("cpu")[elt][0].detach()
                    dec_decoded = output.to("cpu")[elt][0].detach()
                    dec_mask = mask.to("cpu")[elt][0].detach()
                    dec_reconstructed = reconstructed.to("cpu")[elt][0].detach()
                    dec_clas_reconstructed = reconstructed.to("cpu")[elt][
                        1
                    ].detach()
                    display_images(
                       [
                           dec_for_train,
                           dec_clas_train,
                           dec_mask,
                           dec_for_autoencoder,
                           dec_decoded,
                           dec_reconstructed,
                           dec_clas_reconstructed,
                       ],
                       to_file=elt,
                    )

        avg_loss = total_loss / dataset_per_epoch
        scheduler.step(avg_loss)

        print(
            f"Epoch {i + 1} | Loss {total_loss / (dataset_len / batch_size):.5f} (Saved)"
        )

        if total_loss < 0:
            raise Exception("Explosion - self terminating")

        model.checkpoint(checkpoint_path)
