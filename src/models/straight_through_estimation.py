import torch
from torch import nn

# TODO:
class StraightThroughEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, function):
        ctx.save_for_backward(*input)
        with torch.no_grad():
            output = function(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None



class STELayer(nn.Module):
    def forward(self, input, function):
        return function(input)

    def backward(self, grad_output):
        return grad_output

