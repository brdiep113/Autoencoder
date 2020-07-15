from model import PointDetectorNet
from torchsummary import summary
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from utils.loss import ocdnet_loss

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'

def train_net(net, epochs=5, img_scale, val_percent, batch_size):
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    criterion = ocdnet_loss()
    loss_position = nn.BCELoss()
    loss_p = None

    for epoch in range(epochs):
        net.train()
        loss = criterion()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = PointDetectorNet()

    net.to(device=device)
    train_net(net)

    print(summary(net, (3, 1024, 1024)))