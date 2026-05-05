from torch import nn, cat, linspace, clamp, no_grad
from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange
from .embeddings import ContinuousEmbedding, DiscreteEmbedding

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

class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, dilation=2, padding=2)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, dilation=2, padding=2)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x):
        x = x # + embeddings[:, : x.shape[0], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x


class AddCoords(nn.Module):
    def __init__(self, height, width):
        super().__init__()

        xx_range = linspace(0, 1, steps=width)
        yy_range = linspace(0, 1, steps=height)

        xx_channel = xx_range.view(1, 1, 1, width).expand(1, 1, height, width)
        yy_channel = yy_range.view(1, 1, height, 1).expand(1, 1, height, width)

        grid = cat([xx_channel, yy_channel], dim=1)
        self.register_buffer('grid', grid)

    def forward(self, x):
        batch_size = x.size(0)
        return self.grid.expand(batch_size, -1, -1, -1)


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mask_conv = nn.Conv2d(1, 1,
                                   self.kernel_size, self.stride, self.padding,
                                   self.dilation, bias=False)

        nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # 1. Standard convolution on MASKED input
        # This prevents 'leaking' padded zeros or invalid data into the sum
        output = super().forward(input * mask)

        with no_grad():
            # 2. Count valid pixels in the window
            mask_count = self.mask_conv(mask)
            
            # 3. Calculate compensation ratio
            # win_size is the total possible pixels (e.g., 9 for a 3x3 kernel)
            win_size = self.kernel_size[0] * self.kernel_size[1]
            
            # If mask_count is 0, the ratio is 0 (to avoid div by zero)
            # If mask_count > 0, we scale the output
            mask_ratio = win_size / (mask_count + 1e-8)
            
            # 4. Create the mask for the NEXT layer
            # Any pixel where mask_count > 0 becomes a 1
            new_mask = clamp(mask_count, 0, 1)

        # 5. Apply scaling and bias (bias must be handled separately in PConv)
        # We multiply by new_mask to ensure invalid regions stay 0
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = (output - bias_view) * mask_ratio + bias_view
            output = output * new_mask
        else:
            output = output * mask_ratio * new_mask

        return output, new_mask

class UnetLayer(nn.Module):
    def __init__(
        self,
        upscale: bool,
        attention: bool,
        num_groups: int,
        dropout_prob: float,
        num_heads: int,
        C: int,
    ):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(
                C, C // 2, kernel_size=4, stride=2, padding=1
            )
        else:
            self.conv = nn.Conv2d(C, C * 2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(
                C, num_heads=num_heads, dropout_prob=dropout_prob
            )

    def forward(self, x):
        x = self.ResBlock1(x)
        if hasattr(self, "attention_layer"):
            x = self.attention_layer(x)
        x = self.ResBlock2(x)
        return self.conv(x), x

class UNET(nn.Module):
    def __init__(
        self,
        Channels = [32, 64, 128, 256, 128, 64],
        Attentions = [False, False, False, True, False, False],
        Upscales = [ False, False, False, True, True, True],
        num_groups: int = 16,
        dropout_prob: float = 0.05,
        num_heads: int = 8,
        input_channels: int = 2,
        output_channels: int = 1,
    ):
        super().__init__()
        self.coords = AddCoords(256, 256)
        self.num_layers = len(Channels)

        # 3 categories (Unknown | Building | Terrain)
        self.discrete_embeddings = DiscreteEmbedding(3, 8)
        self.continuous_embeddings = ContinuousEmbedding(64, 16)

        self.shallow_conv = PartialConv2d(
            input_channels + 8 + 16 + 2, Channels[0], kernel_size=17, dilation=4, padding=32
        )

        out_channels = (Channels[-1] // 2)
        #out_channels = (Channels[-1] // 2) + (Channels[0])

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

        self.late_conv = nn.Conv2d(
            out_channels, out_channels // 2, kernel_size=3, padding=1
        )

        self.output_conv = nn.Conv2d(out_channels // 2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data, mask):

        mask = mask.float()

        continuous = self.continuous_embeddings((data[:,0,:,:]))
        discrete = self.discrete_embeddings(data[:,1,:,:].long())
        coords = self.coords(data)
        combined = cat([data, continuous, discrete, coords], dim=1)
        data, mask= self.shallow_conv(combined, mask)

        residuals = []

        for i in range(self.num_layers // 2):
            layer = getattr(self, f"Layer{i + 1}")
            data, r = layer(data)
            residuals.append(r)

        for i in range(self.num_layers // 2, self.num_layers):
            layer = getattr(self, f"Layer{i + 1}")
            data, _r = layer(data)
            #x = concat(
            #    (x, residuals[self.num_layers - i - 1]), dim=1
            #)

        data = self.late_conv(data)
        data = self.relu(data)
        return self.output_conv(data)
