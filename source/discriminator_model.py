import torch

from torch import nn
from torch.nn.utils import weight_norm
from torch.nn.utils import spectral_norm

from source.utils import calc_padding, init_weights


class PeriodDiscriminator(torch.nn.Module):
    def __init__(self, period: int, norm_f):
        super().__init__()
        self.period = period
        self.activation = torch.nn.LeakyReLU(0.1)
        kernel_size = 5
        stride = 3

        self.layers = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(calc_padding(5, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(calc_padding(5, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(calc_padding(5, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(calc_padding(5, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.post_conv = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        T = x.size(1)
        if T % self.period:
            x = nn.functional.pad(x, (0, self.period - T % self.period), "reflect")
        channels = T // self.period + (1 if T % self.period else 0)
        x = x.view(-1, 1, channels, self.period)

        layer_acts = []
        for layer in self.layers:
            x = self.activation(layer(x))
            layer_acts.append(x.flatten(1))
        x = self.post_conv(x)
        layer_acts.append(x.flatten(1))

        return x.flatten(1, -1), torch.cat(layer_acts, 1)


class ScaleDiscriminator(torch.nn.Module):
    def __init__(self, norm_f):
        super().__init__()
        self.activation = torch.nn.LeakyReLU(0.1)

        self.layers = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.post_conv = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))
        
    def forward(self, x):
        x = x.unsqueeze(1)

        layer_acts = []
        for layer in self.layers:
            x = self.activation(layer(x))
            layer_acts.append(x.flatten(1))
        x = self.post_conv(x)
        layer_acts.append(x.flatten(1))

        return x.flatten(1, -1), torch.cat(layer_acts, 1)


class SuperDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        disc_periods = (2, 3, 5, 7, 11)
        self.mpd_discriminators = torch.nn.ModuleList([
            PeriodDiscriminator(p, weight_norm) for p in disc_periods
        ])
        self.msd_discriminators = torch.nn.ModuleList([
            ScaleDiscriminator(spectral_norm),
            ScaleDiscriminator(weight_norm),
            ScaleDiscriminator(weight_norm),
        ])

        self.post_pooling = torch.nn.AvgPool1d(4, 2, 2)

        self.apply(init_weights)

    def forward(self, x):
        preds = []
        layer_acts = []

        for model in self.mpd_discriminators:
            pred, hidden = model(x)
            preds.append(pred)
            layer_acts.append(hidden)
        
        for model in self.msd_discriminators:
            pred, hidden = model(x)
            preds.append(pred)
            layer_acts.append(hidden)
            x = self.post_pooling(x)

        return torch.cat(preds, 1), torch.cat(layer_acts, 1)