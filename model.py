import torch
import torch.nn as nn
import torch.nn.functional as F
from model_layers import *


class PointDetectorNet(nn.Module):
    def __init__(self):
        super(PointDetectorNet, self).__init__()

        # Encoder Backbone
        self.vgg1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.vgg2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.vgg3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.vgg4 = DoubleConv(128, 256)

        # Score Module
        self.score = ScoreHead(256, 256)

        # Location Module
        self.location = LocationHead(256, 256)
        self.pixel_shuffle = nn.PixelShuffle(8)

        # Descriptor Module
        self.descriptor1 = DescriptorHeadA(256, 512, 256)
        self.descriptor2 = DescriptorHeadB(256, 256)

    def forward(self, x):
        x = self.vgg1(x)
        x = self.pool1(x)
        x = self.vgg2(x)
        x = self.pool2(x)
        x3 = self.vgg3(x)
        x = self.pool3(x3)
        x = self.vgg4(x)

        score = self.score(x)
        location = self.location(x)
        descriptor = self.descriptor1(x)
        descriptor = self.descriptor2(descriptor, x3)

        return score, location, descriptor

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()

        # Encoder Backbone
        self.vgg1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.predict1 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.vgg2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.predict2 = nn.Conv2d(128, 2, kernel_size=3, padding=1)
        self.vgg3 = TripleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.predict3 = nn.Conv2d(256, 2, kernel_size=3, padding=1)
        self.vgg4 = TripleConv(256, 512)

        # Decoder
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.out = nn.Conv2d(32, 2, kernel_size=1)


    def forward(self, x):
        x1 = self.vgg1(x)
        x2 = self.pool1(x1)
        p1 = self.predict1(x2)
        x3 = self.vgg2(x2)
        x4 = self.pool2(x3)
        p2 = self.predict2(x4)
        x5 = self.vgg3(x4)
        x6 = self.pool3(x5)
        p3 = self.predict3(x6)
        x7 = self.vgg4(x6)
        x = self.up1(x7, p3)
        x = self.up2(x, p2)
        x = self.up3(x, p1)
        logits = self.outc(x)
        return logits