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
parser.add_argument('--model_name', default='resnet50', type=str, help='model name. vgg16 or resnet20')
parser.add_argument('--merge', default=True, type=bool, help='merge conv and bn')
parser.add_argument('--train_batch', default=128, type=int, help='batch size for get max')
parser.add_argument('--batch_num', default=1, type=int, help='number of train batch')
parser.add_argument('--spicalib', default=0, type=int, help='allowance for spicalib')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for testing')
parser.add_argument('--seed', default=42, type=int, help='seed')
args = parser.parse_args()


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += identity
        x = self.relu3(x)
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
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)



    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        identity = self.shortcut(identity)
        identity = self.bn_shortcut(identity)

        #print(x.shape)
        #print(identity.shape)

        x += identity
        #print("x += identity的形状：",x.shape)
        x = self.relu3(x)
        #print("x = self.relu(x)的形状：", x.shape)
        #print('通过')
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
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
        x = self.relu0(x)
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


    if args.model_name == 'resnet50':
        net =  ResNet50()
        net.load_state_dict(torch.load('./preliminary_study_of_convertor/pretrained_models/Resnet50/CIFAR10_Resnet50.pth', map_location=device))

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

    #89    95