import os
import sys
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import scipy as sp
import scipy.stats
# import matplotlib as mpl
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def show_six_images(tensor):
    # tensor : 6 * C * H * W

    mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
    std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
    denomalization = Denormalize(mean_pix, std_pix)
    transform = transforms.Compose([denomalization, transforms.ToPILImage()])
    img_list = []
    for i in range(6):
        img_list.append(transform(tensor[i]))

    for i in range(1, 7):
        plt.subplot(2, 3, i)
        plt.imshow(img_list[i-1])
    plt.show()


def show_img(img):
    """
    Show an image or images of numpy or ndarray type
    """
    if type(img) is not np.ndarray:
        img = img.permute(1, 2, 0).numpy()
    cv2.imshow('a', img)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()
    else:
        sys.exit()



def get_img_map(root_dir):
    """
    loader images from a root_dir, it consists of some subdirs,
    whose names are labels of classes and contents are the corresponding image data.
    the names of image data should be intergers and are just the ids of them.
    """
    img_map = {}
    reverse_map = {}
    img_list = []
    cls_map = {}
    cls_reverse_map = {}
    cnt = 0
    for sub_dir in os.listdir(root_dir):

        if cls_map.get(sub_dir) is None:
            cls_map[sub_dir] = cnt
            cls_reverse_map[cnt] = sub_dir
            cnt += 1

        for img_name in os.listdir(os.path.join(root_dir, sub_dir)):
            img_path = os.path.join(sub_dir, img_name)
            img_list.append(img_path)
            img_map[img_path] = cls_map[sub_dir]
            if cls_map[sub_dir] not in reverse_map.keys():
                reverse_map[cls_map[sub_dir]] = []
            reverse_map[cls_map[sub_dir]].append(img_path)

    return img_map, reverse_map, img_list, cls_map, cls_reverse_map

def resize_pad(img_np, des_size):
    ratio_src = 1.0 * img_np.shape[0] / img_np.shape[1]
    ratio_des = 1.0 * des_size[0] / des_size[1]
    if ratio_src > ratio_des:
        scale = 1.0 * des_size[0] / img_np.shape[0]
    else:
        scale = 1.0 * des_size[1] / img_np.shape[1]
    img_np = cv2.resize(img_np, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if ratio_src > ratio_des:
        delta = des_size[1]-img_np.shape[1]
        pad = (0, 0, delta//2, delta-delta//2)
    else:
        delta = des_size[0]-img_np.shape[0]
        pad = (delta//2, delta-delta//2, 0, 0)
    img_np = cv2.copyMakeBorder(img_np, pad[0], pad[1], pad[2], pad[3],
                                cv2.BORDER_CONSTANT, value=(0,0,0))
    return img_np

def resize_short(img, des_size=(84, 84)):
    ratio_src = 1.0 * img.size[0] / img.size[1]
    ratio_des = 1.0 * des_size[0] / des_size[1]

    if ratio_src < ratio_des:
        scale = 1.0 * des_size[0] / img.size[0]
    else:
        scale = 1.0 * des_size[1] / img.size[1]
    # img_np = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), resample=Image.LANCZOS)
    return img

def img_list2numpy(img_list, img_size):
    assert(len(img_size) == 2)
    res = []
    for img_path in img_list:
        img = Image.open(img_path)
        img = resize_short(img, img_size)
        # im = im.resize((84, 84), resample=Image.LANCZOS)
        # img_np = np.array(im)

        # img_np = cv2.imread(img_path)
        # img_np = resize_short(img_np, img_size)
        # img_np = np.expand_dims(img_np, 0)
        res.append(img)
    # res = np.concatenate(res, axis=0)
    return res

class DataCollector:
    """
    A data collector which receive sequence of map from str to float and restore them.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = {}

    def update(self, data):
        assert type(data) is dict
        for key, value in data.items():
            if self.values.get(key) is None:
                self.values[key] = []
            self.values[key].append(value)

    def confidence_interval(self, confidence=0.95):
        res = {}
        for key, value in self.values.items():
            a = 1.0 * np.array(value)
            n = len(a)
            m, se = np.mean(a), sp.stats.sem(a)
            h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
            res[key] = h
        return res

    def average(self):
        res = {}
        for key, value in self.values.items():
            res[key] = float(np.array(value).mean())

        return res

    def std(self):
        res = {}
        for key, value in self.values.items():
            res[key] = float(np.array(value).std())

        return res

