import torch
import torch.nn as nn
import matplotlib
matplotlib.use('agg')
import numpy as np
from tqdm import tqdm
from braincog.utils import setup_seed
from braincog.datasets.datasets import get_cifar10_data
from braincog.base.conversion import Convertor
import argparse


parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--T', default=64, type=int, help='simulation time')
parser.add_argument('--p', default=0.99, type=float, help='percentile for data normalization. 0-1')
parser.add_argument('--gamma', default=5, type=int, help='burst spike and max spikes IF can emit')
parser.add_argument('--channelnorm', default=False, type=bool, help='use channel norm')
parser.add_argument('--lipool', default=True, type=bool, help='LIPooling')
parser.add_argument('--smode', default=True, type=bool, help='replace ReLU to IF')
parser.add_argument('--soft_mode', default=True, type=bool, help='soft reset or not')
parser.add_argument('--device', default='4', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda.')
parser.add_argument('--model_name', default='resnet101', type=str, help='model name. vgg16 or resnet20')
parser.add_argument('--merge', default=True, type=bool, help='merge conv and bn')
parser.add_argument('--train_batch', default=128, type=int, help='batch size for get max')
parser.add_argument('--batch_num', default=1, type=int, help='number of train batch')
parser.add_argument('--spicalib', default=0, type=int, help='allowance for spicalib')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for testing')
parser.add_argument('--seed', default=42, type=int, help='seed')
args = parser.parse_args()

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += self.shortcut(residual)
        x = self.relu2(x)

        return x

# 定义ResNet-101主干结构
class ResNet101(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet101, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def evaluate_snn(test_iter, snn, device=None, duration=50):
    accs = []
    snn.eval()

    for ind, (test_x, test_y) in tqdm(enumerate(test_iter)):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        n = test_y.shape[0]
        out = 0
        with torch.no_grad():
            snn.reset()
            acc = []
            for t in range(duration):
                out += snn(test_x)
                #print(out)
                result = torch.max(out, 1).indices
                result = result.to(device)
                acc_sum = (result == test_y).float().sum().item()
                acc.append(acc_sum / n)

        accs.append(np.array(acc))
    accs = np.array(accs).mean(axis=0)

    i, show_step = 1, []
    while 2 ** i <= duration:
        show_step.append(2 ** i - 1)
        i = i + 1

    for iii in show_step:
        print("timestep", str(iii).zfill(3) + ':', accs[iii])
    print("best acc: ", max(accs))


if __name__ == '__main__':
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    setup_seed(seed=args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:%s" % args.device) if args.cuda else 'cpu'

    train_iter, _, _, _ = get_cifar10_data(args.train_batch, same_da=True)
    _, test_iter, _, _ = get_cifar10_data(args.batch_size, same_da=True)


    if args.model_name == 'resnet101':
        net = ResNet101(BasicBlock, [3, 4, 23, 3])
        net.load_state_dict(torch.load('./preliminary_study_of_convertor/pretrained_models/Resnet101/CIFAR10_Resnet101.pth', map_location=device))

    net.eval()
    net = net.to(device)

    converter = Convertor(dataloader=train_iter,
                          device=device,
                          p=args.p,
                          channelnorm=args.channelnorm,
                          lipool=args.lipool,
                          gamma=args.gamma,
                          soft_mode=args.soft_mode,
                          merge=args.merge,
                          batch_num=args.batch_num,
                          spicalib=args.spicalib
                          )
    snn = converter(net)
    print('转换完成')

    evaluate_snn(test_iter, snn, device, duration=args.T)
    print('')

    #0.88  0.92
