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
        self.vgg1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.vgg2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.vgg3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.vgg4 = DoubleConv(128, 256)