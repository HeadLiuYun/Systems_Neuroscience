import numpy as np
import os
import imageio


def get_binary_img(path, new_path):
    n = len(os.listdir(path))
    img = 0
    for i in range(n):
        img_path = os.path.join(path, '%04d.tif' % i)
        save_path = os.path.join(new_path, '%04d.tif' % i)
        temp = np.array(imageio.imread(img_path), dtype=np.float32)
        temp[temp < 0.5] = 0
        temp[temp >= 0.5] = 1
        imageio.imwrite(save_path, temp)
    return img


if __name__ == '__main__':
    ann_path = '../output/Lucchi_ANN/'
    snn_path = '../output/Lucchi_SNN/'
    ann_new_path = '../output/Lucchi_ANN_binary/'
    snn_new_path = '../output/Lucchi_SNN_binary/'
    if not os.path.exists(ann_new_path):
        os.makedirs(ann_new_path)
        if not os.path.exists(snn_new_path):
            os.makedirs(snn_new_path)
    get_binary_img(ann_path, ann_new_path)
    get_binary_img(snn_path, snn_new_path)
