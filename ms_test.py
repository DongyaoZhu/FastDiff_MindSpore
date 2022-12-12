import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.context.set_context(mode=ms.context.PYNATIVE_MODE)
import numpy as np
np.random.seed(0)
in_channels = 3
out_channels = 7
x = np.random.random([1, in_channels, 256]).astype(np.float32)
w0 = np.random.random([out_channels, in_channels, 3]).astype(np.float32)
w1 = np.random.random([out_channels, out_channels, 3]).astype(np.float32)
w0 = ms.Tensor(w0)
w1 = ms.Tensor(w1)


class DiffusionDBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = nn.Conv1d(in_channels, out_channels, 1, has_bias=False, weight_init=w0[:, :, :1])
        self.conv = nn.SequentialCell(
            nn.Conv1d(in_channels, out_channels, 3, has_bias=False, dilation=1, pad_mode='pad', padding=1, weight_init=w0),
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
        print('ms res:', res.shape, res)
        x = ops.expand_dims(x, 3)
        x = ops.interpolate(x, sizes=(size, 1), mode='bilinear', coordinate_transformation_mode='half_pixel')
        x = ops.squeeze(x, 3)
        print('ms x:', x.shape, x)
        for layer in self.conv:
            x = self.act(x)
            x = layer(x)
            print('layer x:', x.shape, x)

        return x


class TimeAware_LVCBlock(nn.Cells):
    ''' time-aware location-variable convolutions
    '''
    def __init__(self,
                 in_channels,
                 cond_channels,
                 upsample_ratio,
                 conv_layers=4,
                 conv_kernel_size=3,
                 cond_hop_length=256,
                 kpnet_hidden_channels=64,
                 kpnet_conv_size=3,
                 kpnet_dropout=0.0,
                 noise_scale_embed_dim_out=512
                 ):
        super().__init__()

        self.cond_hop_length = cond_hop_length
        self.conv_layers = conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.convs = torch.nn.ModuleList()

        self.upsample = torch.nn.ConvTranspose1d(in_channels, in_channels,
                                    kernel_size=upsample_ratio*2, stride=upsample_ratio,
                                    padding=upsample_ratio // 2 + upsample_ratio % 2,
                                    output_padding=upsample_ratio % 2)


        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels,
            conv_layers=conv_layers,
            conv_kernel_size=conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            kpnet_dropout=kpnet_dropout
        )

        # the layer-specific fc for noise scale embedding
        self.fc_t = torch.nn.Linear(noise_scale_embed_dim_out, cond_channels)

        for i in range(conv_layers):
            padding = (3 ** i) * int((conv_kernel_size - 1) / 2)
            conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=conv_kernel_size, padding=padding, dilation=3 ** i)

            self.convs.append(conv)


    def forward(self, data):
        ''' forward propagation of the time-aware location-variable convolutions.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length)
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)

        Returns:
            Tensor: the output sequence (batch, in_channels, in_length)
        '''
        x, audio_down, c, noise_embedding = data
        batch, in_channels, in_length = x.shape

        noise = (self.fc_t(noise_embedding)).unsqueeze(-1)  # (B, 80)
        condition = c + noise  # (B, 80, T)
        kernels, bias = self.kernel_predictor(condition)
        x = F.leaky_relu(x, 0.2)
        x = self.upsample(x)

        for i in range(self.conv_layers):
            x += audio_down
            y = F.leaky_relu(x, 0.2)
            y = self.convs[i](y)
            y = F.leaky_relu(y, 0.2)

            k = kernels[:, i, :, :, :, :]
            b = bias[:, i, :, :]
            y = self.location_variable_convolution(y, k, b, 1, self.cond_hop_length)
            x = x + torch.sigmoid(y[:, :in_channels, :]) * torch.tanh(y[:, in_channels:, :])
        return x

    def location_variable_convolution(self, x, kernel, bias, dilation, hop_size):
        ''' perform location-variable convolution operation on the input sequence (x) using the local convolution kernl.
        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length).
            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length)
            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length)
            dilation (int): the dilation of convolution.
            hop_size (int): the hop_size of the conditioning sequence.
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).
        '''
        batch, in_channels, in_length = x.shape
        batch, in_channels, out_channels, kernel_size, kernel_length = kernel.shape


        assert in_length == (kernel_length * hop_size), "length of (x, kernel) is not matched"

        padding = dilation * int((kernel_size - 1) / 2)
        x = F.pad(x, (padding, padding), 'constant', 0)  # (batch, in_channels, in_length + 2*padding)
        x = x.unfold(2, hop_size + 2 * padding, hop_size)  # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            x = F.pad(x, (0, dilation), 'constant', 0)
        x = x.unfold(3, dilation,
                     dilation)  # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x[:, :, :, :, :hop_size]
        x = x.transpose(3, 4)  # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)
        x = x.unfold(4, kernel_size, 1)  # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        o = torch.einsum('bildsk,biokl->bolsd', x, kernel)
        o = o + bias.unsqueeze(-1).unsqueeze(-1)
        o = o.contiguous().view(batch, out_channels, -1)
        return o


net = DiffusionDBlock(in_channels, out_channels, 4)
net(ms.Tensor(x))


