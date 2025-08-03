from torch import nn
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention


class Attention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C * 3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj1(x)
        x = rearrange(x, "b L (C H K) -> K b H L C", K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        x = scaled_dot_product_attention(
            q, k, v, is_causal=False, dropout_p=self.dropout_prob
        )
        x = rearrange(x, "b H (h w) C -> b h w (C H)", h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, "b h w C -> b C h w")
