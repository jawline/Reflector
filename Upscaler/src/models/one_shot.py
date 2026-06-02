import torch
from torch import no_grad, save, nn, logical_not
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from .unet import Net
from monai.losses import SSIMLoss, MaskedLoss


class Model:
    def __init__(self, lr, device=None, warmup_steps=5000):
        self.model = Net().to(device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        warmup = lambda step: min(1.0, step / warmup_steps)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=warmup)
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
                print("Loaded optimizer")
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
        inference = self.model.forward(data, mask)
        return data + (~mask * inference)

    @no_grad()
    def infer(self, src_frame, src_mask, device=None):
        return self.forward(src_frame, src_mask)

    def train_step(self, data, mask, expected, expected_good_mask):
        active_learning_region = expected_good_mask * logical_not(mask)
        if active_learning_region.sum() == 0:
            print("Zero active region — skipping step")
            return None, torch.zeros(1, device=data.device, requires_grad=False)

        self.optimizer.zero_grad(set_to_none=True)

        data = data * mask

        inferred = self.forward(data, mask)

        huber_loss = self.huber(inferred, expected) * active_learning_region
        huber_loss = huber_loss.sum() / (active_learning_region.sum() + 1e-8)

        ssim_loss = self.ssim(
            inferred, expected, mask=expected_good_mask.float()
        ).squeeze().mean()

        loss = huber_loss * 0.8 + ssim_loss * 0.2

        if not torch.isfinite(loss).item():
            err = []
            for name, val in [("huber", huber_loss), ("ssim", ssim_loss)]:
                if not torch.isfinite(val).all():
                    err.append(name)
            print(f"NaN/Inf loss — skipping ({', '.join(err)} component)")
            return inferred, torch.zeros(1, device=loss.device, requires_grad=False)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return inferred, loss
