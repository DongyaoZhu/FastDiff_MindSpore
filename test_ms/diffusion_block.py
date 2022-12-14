import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.context.set_context(mode=ms.context.PYNATIVE_MODE)
import numpy as np
np.random.seed(0)
in_channels = 32
out_channels = 32
x = np.random.random([1, in_channels, 256]).astype(np.float32)
w0 = np.random.random([out_channels, in_channels, 3]).astype(np.float32)
w1 = np.random.random([out_channels, out_channels, 3]).astype(np.float32)
w0 = ms.Tensor(w0)
w1 = ms.Tensor(w1)


class DiffusionDBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = nn.Conv1d(in_channels, out_channels, 1, has_bias=True, weight_init=w0[:, :, :1])
        self.conv = nn.SequentialCell(
            nn.Conv1d(in_channels, out_channels, 3, has_bias=True, dilation=1, pad_mode='pad', padding=1, weight_init=w0),
            nn.Conv1d(out_channels, out_channels, 3, has_bias=True, dilation=2, pad_mode='pad', padding=2, weight_init=w1),
            nn.Conv1d(out_channels, out_channels, 3, has_bias=True, dilation=4, pad_mode='pad', padding=4, weight_init=w1),
        )
        self.act = nn.LeakyReLU(0.2)
    
    def construct(self, x):
        size = x.shape[2] // self.factor
        res = self.residual_dense(x)
        res = ops.expand_dims(res, 3)
        res = ops.interpolate(res, sizes=(size, 1), mode='bilinear', coordinate_transformation_mode='half_pixel')
        res = ops.squeeze(res, 3)
        # print('ms res:', res.shape, res)
        x = ops.expand_dims(x, 3)
        x = ops.interpolate(x, sizes=(size, 1), mode='bilinear', coordinate_transformation_mode='half_pixel')
        x = ops.squeeze(x, 3)
        # print('ms x:', x.shape, x)
        for layer in self.conv:
            x = self.act(x)
            x = layer(x)
            # print('layer x:', x.shape, x)

        return x


if __name__ == '__main__':
    net = DiffusionDBlock(in_channels, out_channels, 4)
    net(ms.Tensor(x))
