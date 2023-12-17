import torch
from torch import nn
from braincog.utils import setup_seed
from torch.utils.data import Dataset
import PIL.Image as Image
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import time
import imageio
import random


class DoubleConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out


class trainDataset(Dataset):
    def __init__(self, root, transform, label_transform, device):
        self.imgs = self.make_dataset(root)
        self.transform = transform
        self.label_transform = label_transform
        self.device = device

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert("L")
        img_y = Image.open(y_path).convert("L")
        if self.transform is not None:
            random.seed(26)
            torch.manual_seed(26)
            img_x = self.transform(img_x)
        if self.label_transform is not None:
            random.seed(26)
            torch.manual_seed(26)
            img_y = self.label_transform(img_y)
        img_x = img_x.to(self.device)
        img_y = img_y.to(self.device)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

    def make_dataset(self, root):
        imgs = []
        n = len(os.listdir(os.path.join(root, 'images')))
        for i in range(n):
            img = os.path.join(root, 'images', "%04d.png" % i)
            mask = os.path.join(root, 'labels', "%04d.png" % i)
            imgs.append((img, mask))
        return imgs


# 准确率函数
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # torch.reshape(num,-1)
    m2 = target.view(num, -1)  # torch.reshape(num,-1)
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def normalize_tensor(tensor):
    # 获取张量的形状
    shape = tensor.size()

    # 计算归一化的最小值和最大值
    reshaped_tensor = tensor.view(1, 1, -1)
    a = torch.min(reshaped_tensor)
    b = torch.max(reshaped_tensor)
    # 避免分母为零的情况
    if a == b:
        # 如果最小值等于最大值，说明张量的所有元素都相同，直接返回全1的张量
        normalized_tensor = torch.ones_like(reshaped_tensor)
    else:
        # 进行线性归一化
        normalized_tensor = (reshaped_tensor - a) / (b - a)
        normalized_tensor = torch.where(normalized_tensor > 0.5, torch.tensor(1.0), torch.tensor(0.0))

    # 将形状还原回原始形状
    normalized_tensor = normalized_tensor.view(*shape)

    return normalized_tensor


def train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs):
    best = 0
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.BCELoss()
    losses = []

    for epoch in range(num_epochs):
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        losss = []
        train_loss_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            label = y
            l = loss(y_hat, label)  # 损失函数
            losss.append(l.cpu().item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
            train_acc_sum += dice_coeff(y_hat, label).cpu().item()
            n += y.shape[0]  # 记录总图像数
            batch_count += 1  # 记录batch数
        scheduler.step()
        test_acc = evaluate_accuracy(test_iter, net)
        losses.append(np.mean(losss))
        print('epoch %d, lr %.6f,train loss %.6f, train acc %.6f, test acc %.6f, time %.1f sec'
              % (
                  epoch + 1, learning_rate, train_loss_sum / batch_count, train_acc_sum / n, test_acc,
                  time.time() - start))

        if test_acc > best:
            best = test_acc
            torch.save(net.state_dict(), './Unet.pth')


def evaluate_accuracy(data_iter, net, save_path=None):
    test_acc = 0.0
    n = 0
    num = 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()  # 评估模式
            prediction = net(X)
            # prediction = normalize_tensor(prediction)  # 加了规范化
            if save_path != None:
                pre = prediction.cpu().squeeze().numpy()
                tif_path = os.path.join(save_path, '%04d.tif' % num)
                imageio.imwrite(tif_path, pre)
            test_acc += dice_coeff(prediction, y).cpu().item()
            net.train()  # 训练模式
            n += y.shape[0]
            num += 1

    return test_acc / n


if __name__ == '__main__':
    # 确保实验稳定性
    setup_seed(26)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = 'cuda:3'

    x_transforms = transforms.Compose([
        transforms.RandomChoice([transforms.RandomRotation((90, 90)), transforms.RandomRotation((0, 0))]),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((768, 1024), padding=128, padding_mode='reflect'),
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5], [0.5])  # ->[-1,1]
    ])

    y_transforms = transforms.Compose([
        transforms.RandomChoice([transforms.RandomRotation((90, 90)), transforms.RandomRotation((0, 0))]),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((768, 1024), padding=128, padding_mode='reflect'),
        transforms.ToTensor()
    ])

    x_transforms_test = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5], [0.5])  # ->[-1,1]
    ])

    y_transforms_test = transforms.Compose([
        transforms.ToTensor()
    ])

    batch_size = 4
    train_root = '../data/datasets/Lucchi/train/'
    test_root = '../data/datasets/Lucchi/test/'
    train_iter = DataLoader(trainDataset(train_root, x_transforms, y_transforms, device), batch_size=batch_size,
                            shuffle=True)
    test_iter = DataLoader(trainDataset(test_root, x_transforms_test, y_transforms_test, device), batch_size=1,
                           shuffle=False)

    # 训练
    lr, num_epochs = 0.05, 300
    net = Unet()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
    train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs)

    # 预测
    net.load_state_dict(torch.load("./Unet.pth", map_location=device))
    net = net.to(device)
    save_path = '../output/Lucchi_ANN/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test_acc = evaluate_accuracy(test_iter, net, save_path)
    print('test_acc:', test_acc)

"""
test_acc: 0.8805268674185782
"""
