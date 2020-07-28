import torch
from model import PointDetectorNet
from torchsummary import summary
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from utils.dataset import MyDataset
from torch.utils.data import DataLoader, random_split
from utils.loss import ocdnet_loss, descriptor_loss
import matplotlib.pyplot as plt

loss_train = []
loss_val = []


def train_net(net, val_percent=0.1, batch_size=128, lr=0.001, epochs=5):
    dataset = MyDataset('.')
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    for epoch in range(epochs):
        net.train()

        for _, data in enumerate(train_loader):
            img = data['image']
            # score_target = data['score']
            location_target = data['location']
            descriptor_target = data['descriptor']

            img = img.to(device=device, dtype=torch.float32)
            # score_target = score_target.to(device=device)
            location_target = location_target.to(device=device)
            descriptor_target = descriptor_target.to(device=device)

            score_pred, location_pred, descriptor_pred = net(img)

            loss = ocdnet_loss(1, 1, location_pred, location_target,
                             descriptor_pred, descriptor_target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

        # Validation
        if (epoch + 1) % 100 == 0:

            net.eval()
            epoch_loss_val = 0.0
            with torch.no_grad():

                for _, data in enumerate(val_loader):
                    img = data['image']
                    # score_target = data['score']
                    location_target = data['location']
                    descriptor_target = data['descriptor']

                    img = img.to(device=device, dtype=torch.float32)
                    # score_target = score_target.to(device=device)
                    location_target = location_target.to(device=device)
                    descriptor_target = descriptor_target.to(device=device)

                    score_pred, location_pred, descriptor_pred = net(img)

                    # loss calculation
                    loss_pos_val = ocdnet_loss(1, 1, location_pred, location_target,
                             descriptor_pred, descriptor_target)
                    epoch_loss_val += loss_pos_val.item() * img.size(0)

                scheduler.step(epoch_loss_val)

            torch.save(net.state_dict(), 'model_val_{0}_saved.pth'.format(epoch))


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = PointDetectorNet()
    net.to(device=device)
    train_net(net)

    torch.save(net.state_dict(), 'final_model_saved.pth')

    # Plot loss Evolution
    plt.plot(loss_train, label='training loss')
    plt.plot(loss_val, label='validation loss')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print(summary(net, (3, 1024, 1024)))