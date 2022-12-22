import torch

from torch import nn
from torch.nn.utils import weight_norm

from source.utils import calc_padding, init_weights


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, lrelu_slope=0.1, dilations=(1, 3, 5)):
        super().__init__()
        self.res_parts = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(lrelu_slope),
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, calc_padding(kernel_size, dilation), dilation)),
                nn.LeakyReLU(lrelu_slope),
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, calc_padding(kernel_size, 1), 1))
            )
            for dilation in dilations
        ])

    def forward(self, x):
        for res_part in self.res_parts:
            x += res_part(x.clone())
        
        return x


class MRFLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        kernel_sizes = (3, 7, 11)
        dilations = tuple((1, 3, 5) for _ in range(3))
        lrelu_slope = 0.1

        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels, kernel_size, lrelu_slope, dilations)
            for kernel_size, dilations in zip(kernel_sizes, dilations)
        ])

    def forward(self, x):
        return sum(res(x) for res in self.res_blocks) / len(self.res_blocks)


class Generator(nn.Module):
    def __init__(self, in_channels: int = 80):
        super().__init__()

        fl_kernel_size = 7
        kernel_sizes = (16, 16, 4, 4)
        strides = (8, 8, 2, 2)
        channels = 256

        blocks = nn.ModuleList([
            weight_norm(nn.Conv1d(
                in_channels, channels,
                fl_kernel_size,
                padding=fl_kernel_size // 2
            )),
        ])

        for kernel_size, stride in zip(kernel_sizes, strides):
            blocks.append(
                nn.Sequential(
                    weight_norm(nn.ConvTranspose1d(
                        channels, channels // 2, kernel_size, stride, padding=(kernel_size - stride) // 2
                    )),
                    MRFLayer(channels//2)
                )
            )
            channels //= 2

        blocks.append(
            weight_norm(nn.Conv1d(
                channels, 1, fl_kernel_size, padding=fl_kernel_size // 2
            ))
        )
        self.model = nn.Sequential(*blocks)
        self.apply(init_weights)

    def forward(self, x):
        x = self.model(x).squeeze(1)
        return torch.tanh(x)