from model import PointDetectorNet
from torchsummary import summary

if __name__ == "__main__":
    net = PointDetectorNet()
    print(summary(net, (3, 1024, 1024)))