import torch
import torch.nn as nn
import torch.nn.functional as F

class PointDetectorNet(nn.Module):
    def __init__(self):
        super(PointDetectorNet, self).__init__()

        #Encoder Backbone
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2D(2)
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2D(2)
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2D(2)
        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2D(2)

        #Score Module
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self

        #Location Module

        #Descriptor Module

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


