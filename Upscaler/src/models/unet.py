from torch import nn, cat, linspace, clamp, no_grad, ones, ones_like, zeros, sqrt, max, concat
from torch.nn.functional import scaled_dot_product_attention, conv2d, conv_transpose2d, conv2d, pad, one_hot
from einops import rearrange
from .embeddings import ContinuousEmbedding, DiscreteEmbedding


class Attention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C * 3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x, mask): # Accept mask here
        b, c, h, w = x.shape

        if mask is not None:
            # 1. Take only the first channel of the mask [B, 1, H, W]
            # 2. Flatten to [B, 1, L]
            # 3. Add a dimension for heads to get [B, 1, 1, L]
            m = mask[:, :1, :, :] 
            m = rearrange(m, "b c h w -> b c (h w)").unsqueeze(1) 
            # m is now [8, 1, 1, 1024] - this will work!
        else:
            m = None
    
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj1(x)
        x = rearrange(x, "b L (C H K) -> K b H L C", K=3, H=self.num_heads)
        
        q, k, v = x[0], x[1], x[2]
    
        # Pass the mask here
        x = scaled_dot_product_attention(
            q, k, v, 
            attn_mask=m, # Apply the flattened mask
            is_causal=False, 
            dropout_p=self.dropout_prob if self.training else 0
        )
    
        x = rearrange(x, "b H (h w) C -> b h w (C H)", h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, "b h w C -> b C h w")


class AddCoords(nn.Module):
    def __init__(self, height, width):
        super().__init__()

        xx_range = linspace(0, 1, steps=width)
        yy_range = linspace(0, 1, steps=height)

        xx_channel = xx_range.view(1, 1, 1, width).expand(1, 1, height, width)
        yy_channel = yy_range.view(1, 1, height, 1).expand(1, 1, height, width)

        grid = cat([xx_channel, yy_channel], dim=1)
        self.register_buffer("grid", grid)

    def forward(self, x):
        batch_size = x.size(0)
        return self.grid.expand(batch_size, -1, -1, -1)


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Multi-channel mask weight: [Out, In, H, W]
        mask_weight = ones(self.out_channels, self.in_channels, *self.kernel_size)
        self.register_buffer('mask_weight', mask_weight)
        self.max_mask_val = self.in_channels * self.kernel_size[0] * self.kernel_size[1]

    def forward(self, x, mask=None):

        # Use 'replicate' instead of 'reflection' - it's more robust at edges
        p = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        x_padded = pad(x * mask, p, mode='replicate')
        mask_padded = pad(mask, p, mode='constant', value=0)

        with no_grad():
            # padding=0 because we padded manually
            mask_sum = conv2d(mask_padded, self.mask_weight, None, 
                                self.stride, 0, self.dilation, self.groups)
            
            mask_ratio = self.max_mask_val / (mask_sum + 1e-8)
            new_mask = clamp(mask_sum, 0, 1)

        raw_out = conv2d(x_padded, self.weight, None, 
                           self.stride, 0, self.dilation, self.groups)

        output = raw_out * mask_ratio
        
        if self.bias is not None:
            output += self.bias.view(1, self.out_channels, 1, 1)

        return output * new_mask, new_mask

class PartialConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Transpose weight shape: [in_channels, out_channels, k, k]
        mask_weight = ones(self.in_channels, self.out_channels, *self.kernel_size)
        self.register_buffer('mask_weight', mask_weight)

        # In Transpose, the scaling is based on the input channels contributing to the output
        self.max_mask_val = self.in_channels * self.kernel_size[0] * self.kernel_size[1]

    def forward(self, x, mask=None):

        with no_grad():
            mask_sum = conv_transpose2d(mask, self.mask_weight, None,
                                          self.stride, 0, self.output_padding,
                                          self.groups, self.dilation)
            mask_ratio = self.max_mask_val / (mask_sum + 1e-8)
            new_mask = clamp(mask_sum, 0, 1)

        raw_out = conv_transpose2d(x * mask, self.weight, None,
                                     self.stride, 0, self.output_padding,
                                     self.groups, self.dilation)

        output = raw_out * mask_ratio
        if self.bias is not None:
            output += self.bias.view(1, self.out_channels, 1, 1)

        # Manual Cropping
        if self.padding[0] > 0 or self.padding[1] > 0:
            p_h, p_w = self.padding
            output = output[:, :, p_h:-p_h, p_w:-p_w]
            new_mask = new_mask[:, :, p_h:-p_h, p_w:-p_w]

        return output * new_mask, new_mask

class MaskedInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(ones(num_features))
            self.bias = nn.Parameter(zeros(num_features))

    def forward(self, x, mask):
        x = x * mask

        mask_sum = mask.sum(dim=(2, 3), keepdim=True)
        mu = x.sum(dim=(2, 3), keepdim=True) / (mask_sum + 1e-8)

        x_shifted = (x - mu)
        var = (x_shifted ** 2).sum(dim=(2, 3), keepdim=True) / (mask_sum + self.eps)
        x_normed = x_shifted / sqrt(var + self.eps)
        
        # 2. Apply weights and bias
        if self.affine:
            # We MUST multiply the bias by the mask so it stays 0 in the holes
            w = self.weight.view(1, -1, 1, 1)
            b = self.bias.view(1, -1, 1, 1)
            x_normed = (x_normed * w) + b # Only add bias to valid pixels
            
        return x_normed * mask


class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = MaskedInstanceNorm2d(C, affine=True)
        self.norm2 = MaskedInstanceNorm2d(C, affine=True)
        self.conv1 = PartialConv2d(C, C, kernel_size=3, dilation=2, padding=2)
        self.conv2 = PartialConv2d(C, C, kernel_size=3, dilation=2, padding=2)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, inp, mask):
        
        x = inp

        x = self.norm1(x, mask) 
        x = self.relu(x)
        x, mask = self.conv1(x, mask)

        x = self.dropout(x)

        x = self.norm2(x, mask) 
        x = self.relu(x)
        x, mask = self.conv2(x, mask)

        return inp + x, mask

class DownLayer(nn.Module):

    def __init__(
        self,
        num_groups: int,
        dropout_prob: float,
        num_heads: int,
        C: int,
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.r = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.conv = PartialConv2d(C, C * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x, mask):
        residual, residual_mask = self.r(x, mask)
        downsample, mask = self.conv(residual, residual_mask)
        downsample = self.relu(downsample)
        return downsample, mask, residual, residual_mask

class UpLayer(nn.Module):

    def __init__(
        self,
        num_groups: int,
        dropout_prob: float,
        num_heads: int,
        C: int,
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.r = ResBlock(C=C // 2, num_groups=num_groups, dropout_prob=dropout_prob)
        self.conv = PartialConvTranspose2d(C, C // 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x, mask):
        x, mask = self.conv(x, mask)
        x = self.relu(x)
        x, mask = self.r(x, mask)
        return x, mask

class AttentionLayer(nn.Module):


    def __init__(
        self,
        num_groups: int,
        dropout_prob: float,
        num_heads: int,
        C: int,
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.norm = MaskedInstanceNorm2d(C, affine=True)
        self.r = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.a = Attention(C=C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, mask):
        origin = x
        x = self.norm(x, mask)
        x, mask = self.r(x, mask)
        x = self.relu(x)
        x = self.a(x, mask)
        return origin + x, mask


class Net(nn.Module):
    def __init__(
        self,
        Downsamples=[16, 32, 64],
        Upsamples=[128, 128, 64 + 32],
        num_attention : int= 4,
        num_groups: int = 16,
        dropout_prob: float = 0.01,
        num_heads: int = 8,
        input_channels: int = 2,
        output_channels: int = 1,
    ):
        super().__init__()
        self.coords = AddCoords(256, 256)

        self.shallow_conv = PartialConv2d(
            1 + 3 + 2,
            Downsamples[0],
            kernel_size=16,
            dilation=2,
            padding=15,
        )

        self.downsamples = nn.ModuleList([ DownLayer(num_groups=num_groups, dropout_prob=dropout_prob, C=channels, num_heads=num_heads) for channels in Downsamples])
        self.upsamples = nn.ModuleList([ UpLayer(num_groups=num_groups, dropout_prob=dropout_prob, C=channels, num_heads=num_heads) for channels in Upsamples])
        self.attentions = nn.ModuleList([AttentionLayer(num_groups=num_groups, dropout_prob=dropout_prob, C=Upsamples[0], num_heads=num_heads) for _ in range(num_attention)])

        out_channels = Upsamples[-1] // 2

        self.late_conv = PartialConv2d(
            out_channels, out_channels // 2, kernel_size=3, padding=1
        )

        self.output_conv = PartialConv2d(out_channels // 2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data, mask):
        mask = mask.float()

        # Generate a one hot encoding of our class and then express it as [B; C; H; W]
        discrete = one_hot(data[:, 1, :, :].long(), num_classes=3)
        discrete = discrete.permute(0, 3, 1, 2)

        origin = data[:,0:1,:,:]
        origin_mask = mask
        coords = self.coords(data)

        data = cat([origin, discrete, coords], dim=1)

        # Set mask 0s for the non data channels, embeddings
        extra_channels = discrete.shape[1] + coords.shape[1]
        extra_mask = zeros(mask.shape[0], extra_channels, mask.shape[2], mask.shape[3], device=mask.device)
        mask = cat([mask, extra_mask], dim=1)

        data, mask = self.shallow_conv(data, mask)


        residuals = []
        residual_masks = []

        for layer in self.downsamples:
            data, mask, residual, residual_mask = layer(data, mask)
            residuals.append(residual)
            residual_masks.append(residual_mask)

        for layer in self.attentions:
            data, mask = layer(data, mask)

        for i, layer in enumerate(self.upsamples):
            #for residual in residuals:
            #    print("RS", residual.shape)
            if i > 0:
                residual = residuals.pop()
                residual_mask = residual_masks.pop()
                data = concat((data, residual), dim=1)
                mask = concat((mask, residual_mask), dim=1)
            data, mask = layer(data, mask)


        data, mask = self.late_conv(data, mask)
        data = self.relu(data)

        data, mask = self.output_conv(data, mask)
        data = self.relu(data)

        # By adding the residual only for the masked cells we avoid forcing the model to predict zeros for the valid cells.
        return origin + (data * (1 - origin_mask))
