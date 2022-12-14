import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target='CPU')
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

w0 = ms.Tensor(w0)
w1 = ms.Tensor(w1)
w2 = ms.Tensor(w2)
w3 = ms.Tensor(w3)


class KernelPredictor(nn.Cell):
    def __init__(self,
        cond_channels,
        conv_in_channels,
        conv_out_channels,
        num_conv_layers,
        conv_kernel_size=3,
        kpnet_hidden_channels=64,
        kpnet_conv_size=3,
        kpnet_dropout=0.0,
    ):
        super().__init__()

        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.num_conv_layers = num_conv_layers

        l_w = conv_in_channels * conv_out_channels * conv_kernel_size * num_conv_layers
        l_b = conv_out_channels * num_conv_layers

        act = nn.LeakyReLU(0.1)

        padding = (kpnet_conv_size - 1) // 2
        self.input_conv = nn.SequentialCell(
            nn.Conv1d(cond_channels,
                      kpnet_hidden_channels,
                      5,
                      pad_mode='pad',
                      padding=(5 - 1) // 2,
                      has_bias=True,
                      weight_init=w0
            ),
            act,
        )

        self.residual_conv = nn.SequentialCell(
            nn.Dropout(1 - kpnet_dropout),
            nn.Conv1d(kpnet_hidden_channels,
                      kpnet_hidden_channels,
                      kpnet_conv_size,
                      pad_mode='pad',
                      padding=padding,
                      has_bias=True,
                      weight_init=w1
            ),
            act,
            nn.Conv1d(kpnet_hidden_channels,
                      kpnet_hidden_channels,
                      kpnet_conv_size,
                      pad_mode='pad',
                      padding=padding,
                      has_bias=True,
                      weight_init=w1
            ),
            act,
            nn.Dropout(1 - kpnet_dropout),
            nn.Conv1d(kpnet_hidden_channels,
                      kpnet_hidden_channels,
                      kpnet_conv_size,
                      pad_mode='pad',
                      padding=padding,
                      has_bias=True,
                      weight_init=w1
            ),
            act,
            nn.Conv1d(kpnet_hidden_channels,
                      kpnet_hidden_channels,
                      kpnet_conv_size,
                      pad_mode='pad',
                      padding=padding,
                      has_bias=True,
                      weight_init=w1
            ),
            act,
            nn.Dropout(1 - kpnet_dropout),
            nn.Conv1d(kpnet_hidden_channels,
                      kpnet_hidden_channels,
                      kpnet_conv_size,
                      pad_mode='pad',
                      padding=padding,
                      has_bias=True,
                      weight_init=w1
            ),
            act,
            nn.Conv1d(kpnet_hidden_channels,
                      kpnet_hidden_channels,
                      kpnet_conv_size,
                      pad_mode='pad',
                      padding=padding,
                      has_bias=True,
                      weight_init=w1
            ),
            act,
        )

        self.kernel_conv = nn.Conv1d(
            kpnet_hidden_channels,
            l_w,
            kpnet_conv_size,
            pad_mode='pad',
            padding=padding,
            has_bias=True,
            weight_init=w2
        )
        self.bias_conv = nn.Conv1d(
            kpnet_hidden_channels,
            l_b,
            kpnet_conv_size,
            pad_mode='pad',
            padding=padding,
            has_bias=True,
            weight_init=w3
        )

    def construct(self, c):
        batch, cond_channels, cond_length = c.shape

        c = self.input_conv(c)
        c = c + self.residual_conv(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)

        kernels = k.view(
            batch,
            self.num_conv_layers,
            self.conv_in_channels,
            self.conv_out_channels,
            self.conv_kernel_size,
            cond_length
        )

        bias = b.view(
            batch,
            self.num_conv_layers,
            self.conv_out_channels,
            cond_length
        )
        return kernels, bias


if __name__ == '__main__':
    net = KernelPredictor(
        cond_channels=cond_channels,
        conv_in_channels=in_channels,
        conv_out_channels=2 * in_channels,
        num_conv_layers=num_conv_layers,
        conv_kernel_size=conv_kernel_size,
        kpnet_hidden_channels=kpnet_hidden_channels,
        kpnet_conv_size=kpnet_conv_size,
        kpnet_dropout=kpnet_dropout
    )
    net(ms.Tensor(x))

