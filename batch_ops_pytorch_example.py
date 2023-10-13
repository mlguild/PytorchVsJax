import torch
import torch.nn as nn

from batch_ops_pytorch import (
    BatchConv1DLayer,
    BatchConv2DLayer,
    BatchLinearLayer,
)

x = torch.ones(100, 50, 3, 32, 32)
params = torch.empty(100, 64, 3, 3, 3)
bias = torch.zeros(100, 64)
nn.init.xavier_uniform_(params)

layer = BatchConv2DLayer(
    in_channels=3, out_channels=32, stride=1, padding=0, dilation=1
)

out = layer.forward(x=x, weight=params, bias=bias)

print(out.shape)

x = torch.ones(100, 50, 3, 32)
params = torch.empty(100, 64, 3, 3)
bias = torch.zeros(100, 64)
nn.init.xavier_uniform_(params)

layer = BatchConv1DLayer(
    in_channels=3, out_channels=32, stride=1, padding=0, dilation=1
)

out = layer.forward(x=x, weight=params, bias=bias)

print(out.shape)

x = torch.ones(100, 10, 50)
params = torch.empty(100, 50, 75)
bias = torch.zeros(100, 75)
nn.init.xavier_uniform_(params)

layer = BatchLinearLayer()

out = layer.forward(x=x, weight=params, bias=bias)

print(out.shape)
