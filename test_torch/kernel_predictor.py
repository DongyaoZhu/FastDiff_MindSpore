import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d

import numpy as np
np.random.seed(0)

inner_channels = 32
in_channels = inner_channels
cond_channels = 80
num_conv_layers = 4
conv_kernel_size = 3
cond_hop_length = 8
# cond_hop_length = 64
# cond_hop_length = 256
kpnet_hidden_channels = 64
kpnet_conv_size = 3
kpnet_dropout = 0.0

x = np.random.random([1, cond_channels, 256]).astype(np.float32)
w0 = np.random.random([kpnet_hidden_channels, cond_channels, 5]).astype(np.float32)
w1 = np.random.random([kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size]).astype(np.float32)

conv_in_channels = in_channels
conv_out_channels = 2 * in_channels
l_w = conv_in_channels * conv_out_channels * conv_kernel_size * num_conv_layers
l_b = conv_out_channels * num_conv_layers

w2 = np.random.random([l_w, kpnet_hidden_channels, kpnet_conv_size]).astype(np.float32)
w3 = np.random.random([l_b, kpnet_hidden_channels, kpnet_conv_size]).astype(np.float32)


w0 = torch.tensor(w0)
w1 = torch.tensor(w1)
w2 = torch.tensor(w2)
w3 = torch.tensor(w3)


class KernelPredictor(torch.nn.Module):
    ''' Kernel predictor for the time-aware location-variable convolutions
    '''

    def __init__(self,
                 cond_channels,
                 conv_in_channels,
                 conv_out_channels,
                 conv_layers,
                 conv_kernel_size=3,
                 kpnet_hidden_channels=64,
                 kpnet_conv_size=3,
                 kpnet_dropout=0.0,
                 kpnet_nonlinear_activation="LeakyReLU",
                 kpnet_nonlinear_activation_params={"negative_slope": 0.1}
                 ):
        '''
        Args:
            cond_channels (int): number of channel for the conditioning sequence,
            conv_in_channels (int): number of channel for the input sequence,
            conv_out_channels (int): number of channel for the output sequence,
            conv_layers (int):
            kpnet_
        '''
        super().__init__()

        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers

        l_w = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers
        l_b = conv_out_channels * conv_layers

        padding = (kpnet_conv_size - 1) // 2
        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv1d(cond_channels, kpnet_hidden_channels, 5, padding=(5 - 1) // 2, bias=False),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
        )
        self.input_conv[0].weight.data = w0

        self.residual_conv = torch.nn.Sequential(
            torch.nn.Dropout(kpnet_dropout),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=False),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=False),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Dropout(kpnet_dropout),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=False),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=False),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Dropout(kpnet_dropout),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=False),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
            torch.nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=False),
            getattr(torch.nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
        )
        self.residual_conv[1].weight.data = w1
        self.residual_conv[3].weight.data = w1
        self.residual_conv[6].weight.data = w1
        self.residual_conv[8].weight.data = w1
        self.residual_conv[11].weight.data = w1
        self.residual_conv[13].weight.data = w1

        self.kernel_conv = torch.nn.Conv1d(kpnet_hidden_channels, l_w, kpnet_conv_size,
                                           padding=padding, bias=False)
        self.bias_conv = torch.nn.Conv1d(kpnet_hidden_channels, l_b, kpnet_conv_size, padding=padding,
                                         bias=False)
        self.kernel_conv.weight.data = w2
        self.bias_conv.weight.data = w3

    def forward(self, c):
        '''
        Args:
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)
        Returns:
        '''
        batch, cond_channels, cond_length = c.shape

        c = self.input_conv(c)
        c = c + self.residual_conv(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)

        kernels = k.contiguous().view(batch,
                                      self.conv_layers,
                                      self.conv_in_channels,
                                      self.conv_out_channels,
                                      self.conv_kernel_size,
                                      cond_length)

        bias = b.contiguous().view(batch,
                                   self.conv_layers,
                                   self.conv_out_channels,
                                   cond_length)
        return kernels, bias


if __name__ == '__main__':
    net = KernelPredictor(
        cond_channels=cond_channels,
        conv_in_channels=in_channels,
        conv_out_channels=2 * in_channels,
        conv_layers=num_conv_layers,
        conv_kernel_size=conv_kernel_size,
        kpnet_hidden_channels=kpnet_hidden_channels,
        kpnet_conv_size=kpnet_conv_size,
        kpnet_dropout=kpnet_dropout
    )
    kernels, bias = net(torch.tensor(x))
    print('kernel:', kernel.shape, kernel)
    print('bias:', bias.shape, bias)
