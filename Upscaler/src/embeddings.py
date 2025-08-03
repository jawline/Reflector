from torch import nn, arange, zeros, sin, cos, exp
from math import log


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        position = arange(time_steps).unsqueeze(1).float()
        div = exp(arange(0, embed_dim, 2).float() * -(log(10000.0) / embed_dim))
        embeddings = zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = sin(position * div)
        embeddings[:, 1::2] = cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        return self.embeddings[t][:, :, None, None].to(x.device)
