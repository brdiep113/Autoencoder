from model import PointDetectorNet
from torchsummary import summary
from torch import optim
from tqdm import tqdm

def train_net(net, epochs=5):

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    for epoch in range(epochs):
        net.train()

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = PointDetectorNet()

    net.to(device=device)
    train_net(net)

    print(summary(net, (3, 1024, 1024)))