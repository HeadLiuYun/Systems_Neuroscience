import torch
import matplotlib
matplotlib.use('agg')
import numpy as np
from tqdm import tqdm
from braincog.utils import setup_seed
from braincog.datasets.datasets import get_cifar10_data
from braincog.base.conversion import Convertor
import argparse
import clip


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
parser.add_argument('--model_name', default='vgg16', type=str, help='model name. vgg16 or resnet20')
parser.add_argument('--merge', default=True, type=bool, help='merge conv and bn')
parser.add_argument('--train_batch', default=100, type=int, help='batch size for get max')
parser.add_argument('--batch_num', default=1, type=int, help='number of train batch')
parser.add_argument('--spicalib', default=0, type=int, help='allowance for spicalib')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for testing')
parser.add_argument('--seed', default=42, type=int, help='seed')
args = parser.parse_args()

def clip_sim(img_feature, text_feature):
    image_features = img_feature
    text_features = text_feature
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    logit_scale = model.logit_scale
    logit_scale = logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    return logits_per_image.softmax(dim=-1).cpu().numpy()

def evaluate_converted_CLIP(test_iter, snn, device=None, duration=50):
    model, _ = clip.load("RN50", device=device)
    class_sets = ['a picture of an airplane', 'a picture of a car', 'a picture of a bird', 'a picture of a cat',
                  'a picture of a deer', 'a picture of a dog', 'a picture of a frog', 'a picture of a horse',
                  'a picture of a sheep', 'a picture of a truck']
    text_embeddings = clip.tokenize(class_sets).to(device)
    snn.eval()
    Acc = []
    for ind, (test_x, test_y) in tqdm(enumerate(test_iter)):
        if ind<=77:
            test_x = test_x.to(device)
            test_y = test_y.cpu().numpy()
            with torch.no_grad():
                image_features = 0
                for t in range(duration):
                    print(snn(test_x).shape)
                    image_features += snn(test_x)

                image_features = image_features.to(torch.float16)
                text_features = model.encode_text(text_embeddings)
                cnn_sim = clip_sim(image_features, text_features)


            result = np.argmax(cnn_sim, axis=1)
            acc = ((result == test_y).sum()) / test_y.shape[0]
            Acc.append(acc)

    print("CLIP转SNN后在cifa10测试集上的分类准确率为：", np.array(Acc).mean(axis=0))



def evaluate_CLIP(test_iter, snn, device=None):
    model, _ = clip.load("RN50", device=device)
    class_sets = ['a picture of an airplane', 'a picture of a car', 'a picture of a bird', 'a picture of a cat',
                  'a picture of a deer', 'a picture of a dog', 'a picture of a frog', 'a picture of a horse',
                  'a picture of a sheep', 'a picture of a truck']
    text_embeddings = clip.tokenize(class_sets).to(device)
    snn.eval()
    Acc = []
    for ind, (test_x, test_y) in tqdm(enumerate(test_iter)):
        test_x = test_x.to(device)
        test_y = test_y.cpu().numpy()
        with torch.no_grad():
            # 计算图文相似度
            text_features = model.encode_text(text_embeddings)
            image_features_cnn = model.encode_image(test_x).to(torch.float16)
            cnn_sim = clip_sim(image_features_cnn, text_features)
        result = np.argmax(cnn_sim, axis=1)
        acc = ((result == test_y).sum()) / test_y.shape[0]
        Acc.append(acc)

    print("CLIP在cifa10测试集上的分类准确率为：",np.array(Acc).mean(axis=0))




if __name__ == '__main__':
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    setup_seed(seed=args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:%s" % args.device) if args.cuda else 'cpu'

    train_iter, _, _, _ = get_cifar10_data(args.train_batch, same_da=True, CLIP = True)
    _, test_iter, _, _ = get_cifar10_data(args.batch_size, same_da=True, CLIP = True)

    model, _ = clip.load("RN50", device=device)
    CLIP_v_model = (model.visual).eval()
    net = CLIP_v_model.to(device)

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
    snn = snn.to(torch.float32)

    evaluate_converted_CLIP(test_iter, snn, device, duration=args.T)  #0.1   0.57





