import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
import os
import imageio


def segmentation_index(y_pred, y_true):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    confusion = confusion_matrix(y_true, y_pred)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))

    jaccard_index = jaccard_score(y_true, y_pred)
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    return accuracy, jaccard_index, F1_score


def get_label(path):
    n = len(os.listdir(path))
    img = 0
    for i in range(n):
        img_path = os.path.join(path, '%04d.png' % i)
        temp = np.array(imageio.imread(img_path), dtype=np.float32)
        temp[temp < 0.5] = 0
        temp[temp >= 0.5] = 1
        temp = np.expand_dims(temp, 0)  # 增加一个维度
        if i == 0:
            img = temp
        else:
            img = np.concatenate([img, temp])
    return img


def get_predict(path):
    n = len(os.listdir(path))
    img = 0
    for i in range(n):
        img_path = os.path.join(path, '%04d.tif' % i)
        temp = np.array(imageio.imread(img_path), dtype=np.float32)
        temp[temp < 0.5] = 0
        temp[temp >= 0.5] = 1
        temp = np.expand_dims(temp, 0)  # 增加一个维度
        if i == 0:
            img = temp
        else:
            img = np.concatenate([img, temp])
    return img


if __name__ == '__main__':
    label_path = '../data/datasets/Lucchi/test/labels/'
    ann_path = '../output/Lucchi_ANN/'
    snn_path = '../output/Lucchi_SNN/'
    label = get_label(label_path)
    ann = get_predict(ann_path)
    snn = get_predict(snn_path)
    accuracy, jaccard_index, F1_score = segmentation_index(ann, label)
    print('ANN:', 'accuracy:', accuracy, 'jaccard:', jaccard_index, 'F1:', F1_score)
    accuracy, jaccard_index, F1_score = segmentation_index(snn, label)
    print('SNN:', 'accuracy:', accuracy, 'jaccard:', jaccard_index, 'F1:', F1_score)

"""
0.9854711050939078 0.7963022128140586 0.8866016053797363
"""
