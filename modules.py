import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class DiffusionDBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = nn.Conv1d(in_channels, out_channels, 1, has_bias=True)
        self.conv = nn.SequentialCell(
            nn.Conv1d(in_channels, out_channels, 3, has_bias=True, dilation=1, pad_mode='pad', padding=1),
            nn.Conv1d(out_channels, out_channels, 3, has_bias=True, dilation=2, pad_mode='pad', padding=2),
            nn.Conv1d(out_channels, out_channels, 3, has_bias=True, dilation=4, pad_mode='pad', padding=4),
        )
        self.act = nn.LeakyReLU(0.2)
    
    def construct(self, x):
        size = x.shape[2] // self.factor
        res = self.residual_dense(x)

        res = ops.expand_dims(res, 3)
        res = ops.interpolate(res, sizes=(size, 1), mode='bilinear', coordinate_transformation_mode='half_pixel')
        res = ops.squeeze(res, 3)

        x = ops.expand_dims(x, 3)
        x = ops.interpolate(x, sizes=(size, 1), mode='bilinear', coordinate_transformation_mode='half_pixel')
        x = ops.squeeze(x, 3)

        for layer in self.conv:
            x = self.act(x)
            x = layer(x)

        return x


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
            ),
            act,
            nn.Conv1d(kpnet_hidden_channels,
                      kpnet_hidden_channels,
                      kpnet_conv_size,
                      pad_mode='pad',
                      padding=padding,
                      has_bias=True,
            ),
            act,
            nn.Dropout(1 - kpnet_dropout),
            nn.Conv1d(kpnet_hidden_channels,
                      kpnet_hidden_channels,
                      kpnet_conv_size,
                      pad_mode='pad',
                      padding=padding,
                      has_bias=True,
            ),
            act,
            nn.Conv1d(kpnet_hidden_channels,
                      kpnet_hidden_channels,
                      kpnet_conv_size,
                      pad_mode='pad',
                      padding=padding,
                      has_bias=True,
            ),
            act,
            nn.Dropout(1 - kpnet_dropout),
            nn.Conv1d(kpnet_hidden_channels,
                      kpnet_hidden_channels,
                      kpnet_conv_size,
                      pad_mode='pad',
                      padding=padding,
                      has_bias=True,
            ),
            act,
            nn.Conv1d(kpnet_hidden_channels,
                      kpnet_hidden_channels,
                      kpnet_conv_size,
                      pad_mode='pad',
                      padding=padding,
                      has_bias=True,
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
        )
        self.bias_conv = nn.Conv1d(
            kpnet_hidden_channels,
            l_b,
            kpnet_conv_size,
            pad_mode='pad',
            padding=padding,
            has_bias=True,
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


