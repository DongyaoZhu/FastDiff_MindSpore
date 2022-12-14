import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d


import numpy as np
np.random.seed(0)
in_channels = 32
out_channels = 32
x = np.random.random([1, in_channels, 256]).astype(np.float32)
w0 = np.random.random([out_channels, in_channels, 3]).astype(np.float32)
w1 = np.random.random([out_channels, out_channels, 3]).astype(np.float32)


class DiffusionDBlock(nn.Module):
  def __init__(self, input_size, hidden_size, factor):
    super().__init__()
    self.factor = factor
    self.residual_dense = Conv1d(input_size, hidden_size, 1, bias=False)
    self.conv = nn.ModuleList([
        Conv1d(input_size, hidden_size, 3, bias=False, dilation=1, padding=1),
        Conv1d(hidden_size, hidden_size, 3, bias=False, dilation=2, padding=2),
        Conv1d(hidden_size, hidden_size, 3, bias=False, dilation=4, padding=4),
    ])
    self.residual_dense.weight.data = torch.tensor(w0[:, :, :1])
    self.conv[0].weight.data = torch.tensor(w0)
    self.conv[1].weight.data = torch.tensor(w1)
    self.conv[2].weight.data = torch.tensor(w1)

  def forward(self, x):
    size = x.shape[-1] // self.factor

    residual = self.residual_dense(x)
    # print('torch res:', residual.shape, residual)
    residual = F.interpolate(residual, mode='linear', align_corners=False, size=size)
    # print('torch res:', residual.shape, residual)

    x = F.interpolate(x, mode='linear', align_corners=False, size=size)
    # print('torch x:', x.shape, x)
    for layer in self.conv:
      x = F.leaky_relu(x, 0.2)
      x = layer(x)
    #   print('layer x:', x.shape, x)

    return x + residual


if __name__ == '__main__':
    net = DiffusionDBlock(in_channels, out_channels, 4)
    net(torch.tensor(x))