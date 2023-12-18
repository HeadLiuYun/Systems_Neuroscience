import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from braincog.utils import setup_seed
from braincog.datasets.datasets import get_cifar10_data
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += identity
        x = self.relu(x)
        return x


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super(ConvolutionalBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        self.shortcut = nn.Conv2d(in_channels, out_channels[2], kernel_size=1, stride=stride, padding=0)
        self.bn_shortcut = nn.BatchNorm2d(out_channels[2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        identity = self.shortcut(identity)
        identity = self.bn_shortcut(identity)

        x += identity
        #print("x += identity的形状：",x.shape)
        x = self.relu(x)
        #print("x = self.relu(x)的形状：", x.shape)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = ConvolutionalBlock(64, [64, 64, 256], stride=1)
        self.block2 = IdentityBlock(256, [64, 64, 256])
        self.block3 = IdentityBlock(256, [64, 64, 256])

        self.block4 = ConvolutionalBlock(256, [128, 128, 512])
        self.block5 = IdentityBlock(512, [128, 128, 512])
        self.block6 = IdentityBlock(512, [128, 128, 512])
        self.block7 = IdentityBlock(512, [128, 128, 512])

        self.block8 = ConvolutionalBlock(512, [256, 256, 1024])
        self.block9 = IdentityBlock(1024, [256, 256, 1024])
        self.block10 = IdentityBlock(1024, [256, 256, 1024])
        self.block11 = IdentityBlock(1024, [256, 256, 1024])
        self.block12 = IdentityBlock(1024, [256, 256, 1024])
        self.block13 = IdentityBlock(1024, [256, 256, 1024])

        self.block14 = ConvolutionalBlock(1024, [512, 512, 2048])
        self.block15 = IdentityBlock(2048, [512, 512, 2048])
        self.block16 = IdentityBlock(2048, [512, 512, 2048])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)

        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs, losstype='mse'):
    best = 0
    net = net.to(device)
    print("training on ", device)
    if losstype == 'mse':
       loss = torch.nn.MSELoss()
    else:
        loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    losses = []

    for epoch in range(num_epochs):
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        losss = []
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            label = y
            if losstype == 'mse':
                label = F.one_hot(y, 10).float()
            l = loss(y_hat, label)
            losss.append(l.cpu().item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        scheduler.step()
        test_acc = evaluate_accuracy(test_iter, net)
        losses.append(np.mean(losss))
        print('epoch %d, lr %.6f, loss %.6f, train acc %.6f, test acc %.6f, time %.1f sec'
              % (epoch + 1, learning_rate, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

        if test_acc > best:
            best = test_acc
            torch.save(net.state_dict(), './preliminary_study_of_convertor/pretrained_models/Resnet50/CIFAR10_Resnet50.pth')


def evaluate_accuracy(data_iter, net, device=None, only_onebatch=False):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]

            if only_onebatch: break
    return acc_sum / n


if __name__ == '__main__':
    setup_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    batch_size = 128
    train_iter, test_iter, _, _ = get_cifar10_data(batch_size)

    print('dataloader finished')

    lr, num_epochs = 0.05, 500
    net =  ResNet50()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
    train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs, losstype='crossentropy')

    net.load_state_dict(torch.load('./preliminary_study_of_convertor/pretrained_models/Resnet50/CIFAR10_Resnet50.pth', map_location=device))
    net = net.to(device)
    acc = evaluate_accuracy(test_iter, net, device)
    print(acc)
