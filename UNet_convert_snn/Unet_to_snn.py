import sys
import time

sys.path.append('../')
import torch
import matplotlib

matplotlib.use('agg')
import numpy as np
from tqdm import tqdm
from Unet import Unet, trainDataset, dice_coeff
from braincog.utils import setup_seed
from braincog.datasets.datasets import get_cifar10_data
from braincog.base.conversion import Convertor
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import imageio
import time

parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--T', default=256, type=int, help='simulation time')
parser.add_argument('--p', default=0.99, type=float, help='percentile for data normalization. 0-1')
parser.add_argument('--gamma', default=5, type=int, help='burst spike and max spikes IF can emit')
parser.add_argument('--channelnorm', default=False, type=bool, help='use channel norm')
parser.add_argument('--lipool', default=True, type=bool, help='LIPooling')
parser.add_argument('--smode', default=True, type=bool, help='replace ReLU to IF')
parser.add_argument('--soft_mode', default=True, type=bool, help='soft reset or not')
parser.add_argument('--device', default='3', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda.')
parser.add_argument('--model_name', default='Unet', type=str, help='model name. vgg16 or resnet20')
parser.add_argument('--merge', default=True, type=bool, help='merge conv and bn')
parser.add_argument('--train_batch', default=100, type=int, help='batch size for get max')
parser.add_argument('--batch_num', default=1, type=int, help='number of train batch')
parser.add_argument('--spicalib', default=0, type=int, help='allowance for spicalib')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for testing')
parser.add_argument('--seed', default=42, type=int, help='seed')
args = parser.parse_args()


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

    # 将形状还原回原始形状
    normalized_tensor = normalized_tensor.view(*shape)

    return normalized_tensor


def evaluate_snn(test_iter, snn, save_path, device=None, duration=50):
    start = time.time()
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
            # for t in tqdm(range(duration)):
            for t in range(duration):
                out += snn(test_x)
                # 归一化
                result = normalize_tensor(out)
                # result =out

                acc_sum = dice_coeff(result, test_y).cpu().item()
                acc.append(acc_sum / n)

            # 保存最后一代的结果
            result = normalize_tensor(out)
            pre = result.cpu().squeeze().numpy()
            tif_path = os.path.join(save_path, '%04d.tif' % ind)
            imageio.imwrite(tif_path, pre)

        accs.append(np.array(acc))
    # accs 应该是 iternum * duration 沿着axis=0求平均 得到每代的准确率
    accs = np.array(accs).mean(axis=0)

    i, show_step = 1, []
    while 2 ** i <= duration:
        show_step.append(2 ** i - 1)
        i = i + 1

    for iii in show_step:
        print("timestep", str(iii).zfill(3) + ':', accs[iii])
    print("best acc: ", max(accs))
    # 打印最大值的索引
    best_index = np.argmax(accs)
    print("best duration: ", best_index)
    print('Using time : %.1f mins' % ((time.time() - start) / 60.0))


if __name__ == '__main__':
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    setup_seed(seed=args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:%s" % args.device) if args.cuda else 'cpu'

    # train_iter, _, _, _ = get_cifar10_data(args.train_batch, same_da=True)  # 这个为什么要sameda
    # _, test_iter, _, _ = get_cifar10_data(args.batch_size, same_da=True)

    batch_size = 4
    train_root = '../data/datasets/Lucchi/train/'
    test_root = '../data/datasets/Lucchi/test/'
    save_root = '../output/Lucchi_SNN/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    x_transforms_test = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5], [0.5])  # ->[-1,1]
    ])

    y_transforms_test = transforms.Compose([
        transforms.ToTensor()
    ])
    train_iter = DataLoader(trainDataset(train_root, x_transforms_test, y_transforms_test, device),
                            batch_size=batch_size,
                            shuffle=True)
    test_iter = DataLoader(trainDataset(test_root, x_transforms_test, y_transforms_test, device), batch_size=1,
                           shuffle=False)

    net = Unet()
    net.load_state_dict(torch.load("./Unet.pth", map_location=device))
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

    evaluate_snn(test_iter, snn, save_root, device, duration=args.T)

# ANN test_acc: 0.8621542204510082
# SNN best acc: 0.7037782094695352
