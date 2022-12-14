import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d


import numpy as np
np.random.seed(0)


audio_channels = 1
inner_channels = 32
in_channels = inner_channels
cond_channels = 80
# upsample_ratios = [8, 8, 4]
upsample_ratios = 8
lvc_layers_each_block = 4
lvc_kernel_size = 3
kpnet_hidden_channels = 64
kpnet_conv_size = 3
dropout = 0.0
diffusion_step_embed_dim_in = 128
diffusion_step_embed_dim_mid = 512
diffusion_step_embed_dim_out = 512
cond_hop_length = 8


from kernel_predictor import KernelPredictor
w0 = np.random.random([in_channels, in_channels, upsample_ratios * 2]).astype(np.float32)
w1 = np.random.random([cond_channels, diffusion_step_embed_dim_out]).astype(np.float32)
w2 = np.random.random([in_channels, in_channels, kpnet_conv_size]).astype(np.float32)


class TimeAware_LVCBlock(torch.nn.Module):
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
                                    padding=upsample_ratio // 2 + upsample_ratio % 2, bias=False,
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
            conv = torch.nn.Conv1d(in_channels, in_channels, 
                bias=False,
                kernel_size=conv_kernel_size, padding=padding, dilation=3 ** i)
            conv.weight.data = torch.tensor(w2)
            self.convs.append(conv)
        self.upsample.weight.data = torch.tensor(w0)
        self.fc_t.weight.data = torch.tensor(w1)

    def forward(self, data):
        x, audio_down, c, noise_embedding = data
        batch, in_channels, in_length = x.shape

        noise = (self.fc_t(noise_embedding)).unsqueeze(-1)  # (B, 80)
        condition = c + noise  # (B, 80, T)
        # print('condition:', condition.shape, condition)
        kernels, bias = self.kernel_predictor(condition)
        # print('kernels:', kernels.shape, kernels)
        # print('bias:', bias.shape, bias)
        x = F.leaky_relu(x, 0.2)
        x = self.upsample(x)
        # print('x:', x.shape, x)

        for i in range(self.conv_layers):
            x += audio_down
            y = F.leaky_relu(x, 0.2)
            y = self.convs[i](y)
            y = F.leaky_relu(y, 0.2)

            k = kernels[:, i, :, :, :, :]
            b = bias[:, i, :, :]
            y = self.location_variable_convolution(y, k, b, 1, self.cond_hop_length)
            print('y:', y.shape, y)
            return y
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
        print('    before unfold 1:', x.shape)
        # x = x.unfold(2, hop_size + 2 * padding, hop_size)  # (batch, in_channels, kernel_length, hop_size + 2*padding)
        # print('    after unfold 1:', x.shape)
        # '''
        unfold1 = nn.Unfold(
            kernel_size=(1, hop_size + 2 * padding),
            stride=(1, hop_size),
            dilation=(1, 1),
            padding=0
        )
        np.save('x.npy', x.detach().numpy())

        x = torch.unsqueeze(x, 2)
        print('    before unfold 1:', x.shape)
        x = unfold1(x)  # (batch, in_channels, kernel_length, hop_size + 2*padding)
        # x = x.reshape((batch, in_channels, -1, x.shape[-1]))#.transpose(2, 3)
        print('    after unfold 1:', x.shape)
        # '''
        return x

        if hop_size < dilation:
            x = F.pad(x, (0, dilation), 'constant', 0)
        print('    before unfold 2:', x.shape)
        # x: (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x.unfold(3, dilation, dilation)
        print('    after unfold 2:', x.shape)
        x = x[:, :, :, :, :hop_size]
        x = x.transpose(3, 4)  # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)
        print('    before unfold 3:', x.shape)
        x = x.unfold(4, kernel_size, 1)  # (batch, in_channels, kernel_length, dilation, _, kernel_size)
        print('    after unfold 3:', x.shape)

        o = torch.einsum('bildsk,biokl->bolsd', x, kernel)
        o = o + bias.unsqueeze(-1).unsqueeze(-1)
        o = o.contiguous().view(batch, out_channels, -1)
        return o


lvcb = TimeAware_LVCBlock(
    in_channels=inner_channels,
    cond_channels=cond_channels,
    upsample_ratio=upsample_ratios,
    conv_layers=lvc_layers_each_block,
    conv_kernel_size=lvc_kernel_size,
    cond_hop_length=cond_hop_length,
    kpnet_hidden_channels=kpnet_hidden_channels,
    kpnet_conv_size=kpnet_conv_size,
    kpnet_dropout=dropout,
    noise_scale_embed_dim_out=diffusion_step_embed_dim_out
)

x = np.random.random([1, in_channels, 100]).astype(np.float32)
audio_down = np.random.random([1, in_channels, 800]).astype(np.float32)
c = np.random.random([1, cond_channels, 100]).astype(np.float32)
diffusion_step_embed = np.random.random([1, 512]).astype(np.float32)

x = torch.tensor(x)
audio_down = torch.tensor(audio_down)
c = torch.tensor(c)
diffusion_step_embed = torch.tensor(diffusion_step_embed)

t = (x, audio_down, c, diffusion_step_embed)
lvcb(t)