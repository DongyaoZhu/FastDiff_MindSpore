import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target='CPU')
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


class TimeAware_LVCBlock(nn.Cell):
    def __init__(self,
        in_channels=32,
        cond_channels=80,
        upsample_ratio=8,
        num_conv_layers=4,
        conv_kernel_size=3,
        cond_hop_length=256,
        kpnet_hidden_channels=64,
        kpnet_conv_size=3,
        kpnet_dropout=0.0,
        noise_scale_embed_dim_out=512
    ):
        super().__init__()

        self.cond_hop_length = cond_hop_length
        self.num_conv_layers = num_conv_layers
        self.conv_kernel_size = conv_kernel_size
        self.convs = []

        self.upsample = nn.Conv1dTranspose(
            in_channels,
            in_channels,
            kernel_size=upsample_ratio*2,
            stride=upsample_ratio,
            pad_mode='pad',
            padding=upsample_ratio // 2 + upsample_ratio % 2,
            weight_init=ms.Tensor(w0)
            # output_padding=upsample_ratio % 2
        )
        self.act = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels,
            num_conv_layers=num_conv_layers,
            conv_kernel_size=conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            kpnet_dropout=kpnet_dropout
        )

        # the layer-specific fc for noise scale embedding
        self.fc_t = nn.Dense(noise_scale_embed_dim_out, cond_channels, has_bias=True,
                    weight_init=ms.Tensor(w1)
        )

        for i in range(num_conv_layers):
            padding = (3 ** i) * int((conv_kernel_size - 1) / 2)
            conv = nn.Conv1d(in_channels,
                             in_channels,
                             kernel_size=conv_kernel_size,
                             pad_mode='pad',
                             padding=padding,
                             dilation=3 ** i,
                             has_bias=True,
                             weight_init=ms.Tensor(w2)
            )
            self.convs.append(conv)
        self.convs = nn.SequentialCell(self.convs)

        self.einsum = ops.Einsum('bildsk,biokl->bolsd')

    def location_variable_convolution(self, x, kernel, bias, dilation, hop_size):
        batch, in_channels, in_length = x.shape
        batch, in_channels, out_channels, kernel_size, kernel_length = kernel.shape

        padding = dilation * int((kernel_size - 1) / 2)
        x = ops.pad(x, ((0, 0), (0, 0), (padding, padding)))  # (batch, in_channels, in_length + 2*padding)
        unfold1 = nn.Unfold(
            ksizes=(1, 1, hop_size + 2 * padding, 1),
            strides=(1, 1, hop_size, 1),
            rates=(1, 1, 1, 1),
            padding='valid'
        )
        x = ops.expand_dims(x, 2)
        print('    before unfold 1:', x.shape)
        x = unfold1(x)  # (batch, in_channels, kernel_length, hop_size + 2*padding)
        # x = unfold1(dimension=2, size=hop_size + 2 * padding, step=hop_size)  # (batch, in_channels, kernel_length, hop_size + 2*padding)
        x = x.reshape((batch, in_channels, -1, x.shape[-1])).transpose(0, 1, 3, 2)
        print('    after unfold 1:', x.shape)
        return x

        if hop_size < dilation:
            x = ops.pad(x, ((0, 0), (0, 0), (0, dilation)))
        # x: (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        print('    before unfold 2:', x.shape)
        x = x.unfold(3, dilation, dilation)  
        print('    after unfold 2:', x.shape)
        x = x[:, :, :, :, :hop_size]
        x = x.transpose(3, 4)  # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)
        print('    before unfold 3:', x.shape)
        x = x.unfold(4, kernel_size, 1)  # (batch, in_channels, kernel_length, dilation, _, kernel_size)
        print('    after unfold 3:', x.shape)

        o = self.einsum((x, kernel))
        o = o + bias.unsqueeze(-1).unsqueeze(-1)
        o = o.view(batch, out_channels, -1)
        return o

    def construct(self, data):
        x, audio_down, c, noise_embedding = data
        batch, in_channels, in_length = x.shape

        noise = ops.expand_dims(self.fc_t(noise_embedding), 2)  # (B, 80)
        # print('noise:', noise.shape, noise)
        condition = c + noise  # (B, 80, T)
        kernels, bias = self.kernel_predictor(condition)
        x = self.act(x)
        # print('x:', x.shape, x)
        x = self.upsample(x)
        # print('upsample x:', x.shape, x)

        for i in range(self.num_conv_layers):
            x += audio_down
            y = self.act(x)
            y = self.convs[i](y)
            y = self.act(y)

            k = kernels[:, i, :, :, :, :]
            b = bias[:, i, :, :]
            y = self.location_variable_convolution(y, k, b, 1, self.cond_hop_length)
            print('y:', y.shape, y)
            return y
            x = x + self.sigmoid(y[:, :in_channels, :]) * self.tanh(y[:, in_channels:, :])
        return x


lvcb = TimeAware_LVCBlock(
    in_channels=inner_channels,
    cond_channels=cond_channels,
    upsample_ratio=upsample_ratios,
    num_conv_layers=lvc_layers_each_block,
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

x = ms.Tensor(x)
audio_down = ms.Tensor(audio_down)
c = ms.Tensor(c)
diffusion_step_embed = ms.Tensor(diffusion_step_embed)

t = (x, audio_down, c, diffusion_step_embed)
lvcb(t)