import torch
import torch.nn as nn
import DepthwiseAffineGrid
import DepthwiseGridSampler
import numpy as np

class DepthwiseAffineGridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, size):
        ctx.intermediate_results = size
        return DepthwiseAffineGrid.forward(theta, size)
    @staticmethod
    def backward(ctx, grad):
        size = ctx.intermediate_results
        return DepthwiseAffineGrid.backward(grad, size), None

class DepthwiseGridSamplerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        ctx.save_for_backward(input, grid)
        return DepthwiseGridSampler.forward(input, grid)
    @staticmethod
    def backward(ctx, grad):
        input, grid = ctx.saved_variables
        return DepthwiseGridSampler.backward(grad, input, grid)


class ActiveShift2d(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super(ActiveShift2d, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(in_channel, 2).unsqueeze_(-1))
        self.theta.data.uniform_(-1-(2/27), -1+(2/27)) # have to change
        self.base_theta = torch.FloatTensor([[1, 0], [0, 1]]).repeat(in_channel, 1, 1)

    def forward(self, x):
        theta = torch.cat([self.base_theta.type_as(self.theta), self.theta], dim=-1)
        grid = DepthwiseAffineGridFunction.apply(theta, x.size())
        shifted = DepthwiseGridSamplerFunction.apply(x.contiguous(), grid.contiguous())
        return shifted