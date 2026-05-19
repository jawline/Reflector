import torch
from torch import no_grad, save, nn, logical_not
from torch.optim import Adam
from .unet import Net
from monai.losses import SSIMLoss, MaskedLoss


class Model:

    def __init__(self, lr, device=None):
        self.model = Net().to(device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.huber = nn.HuberLoss(reduction="none")
        self.ssim = MaskedLoss(
            loss=SSIMLoss(spatial_dims=2, data_range=1.0, reduction="none")
        )

    def load(self, file, device=None):
        try:
            print("Trying to load", file)
            if file is not None:
                checkpoint = torch.load(file, map_location=device)
                print("Loaded checkpoint")
                self.model.load_state_dict(checkpoint["model"])
                print("Loaded model")
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                print("Loaded all")
        except Exception as e:
            print("Could not load checkpoint", e)

    def checkpoint(self, checkpoint_path):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        print("Saving checkpoint to", checkpoint_path)
        save(checkpoint, checkpoint_path)
        print("Saved checkpoint to", checkpoint_path)

    def forward(self, data, mask):
        # We only care about the model learning the masked region
        return data + ((1 - mask) * self.model.forward(data, mask))

    def infer(self, src_frame, src_mask, device=None):

        with no_grad():
            forward = self.forward(src_frame, src_mask)

        return forward

    def train_step(self, data, mask, expected, expected_good_mask, device=None):
        self.optimizer.zero_grad(set_to_none=True)

        data = data * mask
        inferred = self.forward(data, mask)

        # We only want to predict errors around the regions we aimed to fill in
        training_target_mask = expected_good_mask * logical_not(mask)
        huber_loss = self.huber(inferred, expected)

        # We should not backprop the loss from the regions that were not known in the original input
        huber_loss = huber_loss * training_target_mask

        # Manually take the mean, accounting for missing pixels
        huber_loss = huber_loss.sum() / (training_target_mask.sum() + 1e-8)

        # SSIM loss against the entire ground truth, ignore pixels we did not know about.
        ssim_loss = self.ssim(
            inferred,
            expected,
            mask=expected_good_mask,
        ).mean()

        loss = (huber_loss * 0.8) + (ssim_loss * 0.2)

        loss.backward()

        self.optimizer.step()

        return inferred, loss
