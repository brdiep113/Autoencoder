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
            nn.Dropout(p=0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ScoreHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ScoreHead, self).__init__()
        self.score = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.2, inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.score(x)


class LocationHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(LocationHead, self).__init__()
        self.location = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.2, inplace=True),
            nn.Conv2d(out_channels, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.location(x)


class DescriptorHeadA(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels):
        super(DescriptorHeadA, self).__init__()
        self.descriptor = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.Dropout(p=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.descriptor(x)


class DescriptorHeadB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DescriptorHeadB, self).__init__()
        self.descriptor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.descriptor(x)
