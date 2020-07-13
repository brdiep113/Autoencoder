import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    2 * ( Convolution2D + Batch Normalization + LeakyReLU )
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ScoreModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ScoreModule, self).__init__()
        self.score = nn.Sequential(
            nn.Conv2d(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.5, inplace=True),
            nn.Conv2d(out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.score(x)

class DescriptorModule(nn.Module):
    pass

class LocationModule(nn.Module):
    pass