import torch
from torch import clamp, no_grad, save, nn
from torch.optim import Adam
from .unet import Net
from pytorch_msssim import ssim


class Model:

    def __init__(self, lr, device=None):
        self.model = Net().to(device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.huber = nn.HuberLoss(reduction="none")

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

    def infer(self, src_frame, src_mask, device=None):

        with no_grad():
            forward = self.model.forward((src_frame * src_mask))

        return forward

    def train_step(self, data, mask, expected, expected_good_mask, device=None):
        self.optimizer.zero_grad(set_to_none=True)
        data = data * mask
        inferred = self.model.forward(data, mask)

        huber_loss = self.huber(inferred, expected)
        huber_loss = huber_loss * expected_good_mask # We should not backprop the loss from the regions that were not known in the original input
        huber_loss = huber_loss.sum() / (mask.sum() + 1e-8)

        #ssim_loss = 1 - ssim(
        #    clamp(inferred, min=0, max=1) * expected_good_mask,
        #    clamp(expected, min=0, max=1) * expected_good_mask,
        #    data_range=1.0,
        #    size_average=True,
        #)

        loss = (huber_loss * 1) #+ (ssim_loss * 0.2)

        loss.backward()

        self.optimizer.step()

        return inferred, loss
