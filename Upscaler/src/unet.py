from torch import nn, concat
from embeddings import SinusoidalEmbeddings
from unet_layer import UnetLayer
from typing import List
from masked_grad import MaskedRoot


class UNET(nn.Module):
    def __init__(
        self,
        Channels: List = [64, 128, 256, 512, 512, 384],
        Attentions: List = [False, True, True, True, False, False],
        Upscales: List = [False, False, False, True, True, True],
        num_groups: int = 32,
        dropout_prob: float = 0.1,
        num_heads: int = 8,
        input_channels: int = 1,
        output_channels: int = 1,
        time_steps: int = 1000,
    ):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(
            input_channels, Channels[0], kernel_size=3, padding=1
        )
        out_channels = (Channels[-1] // 2) + Channels[0]
        self.late_conv = nn.Conv2d(
            out_channels, out_channels // 2, kernel_size=3, padding=1
        )
        self.output_conv = nn.Conv2d(out_channels // 2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(
            time_steps=time_steps, embed_dim=max(Channels)
        )
        self.mask = MaskedRoot()
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads,
            )
            setattr(self, f"Layer{i + 1}", layer)

    def forward(self, x, t, mask):
        (x, t) = self.mask((x, t), mask)
        x = self.shallow_conv(x)
        residuals = []
        for i in range(self.num_layers // 2):
            layer = getattr(self, f"Layer{i + 1}")
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings)
            residuals.append(r)
        for i in range(self.num_layers // 2, self.num_layers):
            layer = getattr(self, f"Layer{i + 1}")
            x = concat(
                (layer(x, embeddings)[0], residuals[self.num_layers - i - 1]), dim=1
            )
        return self.output_conv(self.relu(self.late_conv(x)))
