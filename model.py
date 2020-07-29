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
        #self.score = ScoreHead(256, 256)

        # Location Module
        self.location = LocationHead(256, 256)

        # Descriptor Module
        self.descriptor = SingleDescriptorHead(256, 512, 256)
        self.descriptor1 = DescriptorHeadA(256, 512, 256)
        self.descriptor2 = DescriptorHeadB(256, 256)

    def forward(self, x):
        x = self.vgg1(x)
        x = self.pool1(x)
        x = self.vgg2(x)
        x = self.pool2(x)
        x = self.vgg3(x)
        x = self.pool3(x)
        x = self.vgg4(x)

        #score = self.score(x)
        location = self.location(x)
        #descriptor = self.descriptor1(x)
        #descriptor = self.descriptor2(descriptor, x3)
        descriptor = self.descriptor(x)

        return location, descriptor

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


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # four pairs of convolution layers
        self.conv1 = nn.Conv2d(4, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(128)

        # position module
        self.conv9 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(256)

        self.conv10 = nn.Conv2d(256, 8 * 8 + 1, 1, 1, 0)  # 8=scale & 1=dustbin
        self.bn10 = nn.BatchNorm2d(65)

        self.pixel_shuffle = nn.PixelShuffle(8)

        # feature module

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2, stride=2))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(F.max_pool2d(self.bn4(self.conv4(x)), 2, stride=2))

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(F.max_pool2d(self.bn6(self.conv6(x)), 2, stride=2))

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        fm = x
        x = F.relu(self.bn9(self.conv9(x)))
        x = self.bn10(self.conv10(x))

        prob = torch.softmax(x, 1)  # channel-wise softmax
        prob = prob[:, :-1, :, :]  # removes dustbin dim.
        prob = self.pixel_shuffle(prob)
        prob = torch.squeeze(prob, 1)

        return {'logits': x, 'prob': prob, 'latent': fm}
