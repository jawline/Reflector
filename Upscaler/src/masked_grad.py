from torch.autograd import Function
from torch import nn


class MaskedGrad(Function):
    @staticmethod
    def forward(ctx, input_image, nan_mask):
        ctx.save_for_backward((input_image, nan_mask))
        return input_image

    @staticmethod
    def backward(ctx, grad_out):
        input_image, nan_mask = ctx.saved_tensors
        print(grad_out.shape, input_image.shape, nan_mask.shape)
        return (grad_out * input_image) * nan_mask


class MaskedRoot(nn.Module):
    def forward(self, input_image, nan_mask):
        return MaskedGrad.apply(input_image, nan_mask)

    def backward(self, grad_out):
        return MaskedGrad.backward(grad_out)
