from torch import nn, arange, zeros, sin, cos, exp, linspace, randn, tensor, abs, matmul
from torch.nn import functional as F
from math import log


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        position = arange(time_steps).unsqueeze(1).float()
        div = exp(arange(0, embed_dim, 2).float() * -(log(10000.0) / embed_dim))
        embeddings = zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = sin(position * div)
        embeddings[:, 1::2] = cos(position * div)
        self.register_buffer("embeddings", embeddings)

    def forward(self, x, t):
        return self.embeddings[t][:, :, None, None]


class DiscreteEmbedding(nn.Module):
    def __init__(self, num_categories, embedding_dim):
        super().__init__()
        # Maps 0, 1, 2 to an 8-dimensional vector
        self.cat_embed = nn.Embedding(num_categories, embedding_dim)

    def forward(self, cat_map):
        # cat_map shape: [B, H, W]
        # output shape: [B, H, W, 8] -> permute to [B, 8, H, W]
        embedded = self.cat_embed(cat_map).permute(0, 3, 1, 2)
        return embedded


class ContinuousEmbedding(nn.Module):
    def __init__(self, num_bins, embedding_dim, min_val=0.0, max_val=100.0):
        super().__init__()
        # 1. Learnable Bin Centers: Where do the 'categories' live?
        # We initialize them evenly, but the optimizer will move them.
        initial_centers = linspace(min_val, max_val, num_bins)
        self.centers = nn.Parameter(initial_centers)

        # 2. Learnable Weights: These are the actual embedding vectors
        # This replaces nn.Embedding weight matrix
        self.embeddings = nn.Parameter(randn(num_bins, embedding_dim))

        # 3. Learnable Temperature: Controls how "sharp" the binning is
        self.temperature = nn.Parameter(tensor(1.0))

    def forward(self, x):
        # x: (batch_size) -> (batch_size, 1)
        x = x.unsqueeze(-1)

        # Calculate distance between each input and each bin center
        # distances: (batch_size, num_bins)
        distances = abs(x - self.centers)

        # Compute "soft" assignment weights using Softmax
        # Lower distance = higher weight for that bin
        weights = F.softmax(-self.temperature * distances, dim=-1)

        # Multiply weights by the embedding matrix (Weighted Sum)
        # (batch_size, num_bins) @ (num_bins, embedding_dim)
        soft_embedding = matmul(weights, self.embeddings)

        return soft_embedding.permute(0, 3, 1, 2)
