import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import *

class PointDetectorNet(nn.Module):
    def __init__(self):
        super(PointDetectorNet, self).__init__()

        #Encoder Backbone
        self.vgg1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2D(2)
        self.vgg2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2D(2)
        self.vgg3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2D(2)
        self.vgg4 = DoubleConv(128, 256)

        #Score Module
        self.score = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #Location Module
        self.location = 1

        #Descriptor Module
        self.descriptor = 1

    def forward(self, x):
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        pool1 = self.pool1(x)
        x = F.relu(self.conv2a(pool1))
        x = F.relu(self.conv2b(x))
        pool2 = self.pool2(x)
        x = F.relu(self.conv3a(pool2))
        x = F.relu(self.conv3b(x))
        pool3 = self.pool3(x)
        x = F.relu(self.conv4a(pool3))
        x = F.relu(self.conv4b(x))
        x = torch.cat([x, pool3], dim=1)
        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5b(x))


