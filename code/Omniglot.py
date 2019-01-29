import sys
import os
import shutil
import random
import numpy as np
import itertools
import pickle
import cv2
import utils
from PIL import Image

import torch
import torchnet as tnt
from torchvision import transforms

from DataLoader import EpisodeLoader

sys.path.append('../')

_ROOT_DIR = '../datasets/omniglot/data'

def rename_folder(root_dir='../datasets/omniglot/python/images_background',
                  des_dir='../datasets/omniglot/train_data'):
    cnt = 1
    root_dir2 = des_dir 
    for sub_dir in os.listdir(root_dir):
        path1 = os.path.join(root_dir, sub_dir)
        for sub_dir2 in os.listdir(path1):
            path2 = os.path.join(path1, sub_dir2)
            for img_name in os.listdir(path2):
                img_path = os.path.join(path2, img_name)
                img_np = cv2.imread(img_path)
                folder_path = os.path.join(root_dir2, sub_dir + '_' + sub_dir2[-2:])
                for direction in range(4):
                    img_np = np.rot90(img_np)
                    temp = folder_path+'_'+str(direction)
                    if not os.path.exists(temp):
                        os.makedirs(temp)
                    cv2.imwrite(os.path.join(temp, str(cnt)+'.png'), img_np)
                    cnt += 1

def img_list2numpy(img_list, img_size):
    assert(len(img_size) == 2)
    res = []
    for img_path in img_list:
        img = Image.open(img_path)
        img = img.convert('L')
        img = img.resize((28,28), resample=Image.LANCZOS) # per Chelsea's implementation
        img_np = np.array(img, dtype='float32')
        img_np = np.expand_dims(img_np, 2)
        # im = im.resize((84, 84), resample=Image.LANCZOS)
        # img_np = np.array(im)
        # img_np = cv2.imread(img_path)
        # img_np = resize_short(img_np, img_size)
        # img_np = np.expand_dims(img_np, 0)
        res.append(img_np)
    res = np.stack(res, axis=0)
    return res

def gen_train_test_pickle():
    ROOT_DIR = '../datasets/omniglot/all_data'
    _, _, img_dirs, _, _ = utils.get_img_map(ROOT_DIR)
    random.shuffle(img_dirs)
    img_train_dirs = img_dirs[:1200]
    img_lists, cats = [], []
    cat_cnt = 0
    for train_dir in img_train_dirs:
        path1 = os.path.join(ROOT_DIR, train_dir)
        for img_path in os.listdir(path1):
            img_lists.append(os.path.join(path1, img_path))
            cats.append(cat_cnt)
        cat_cnt += 1

    cat_cnt = 0
    img_test_dirs = img_dirs[1200:]
    img_test_lists, test_cats = [], []
    for test_dir in img_test_dirs:
        path1 = os.path.join(ROOT_DIR, test_dir)
        for img_path in os.listdir(path1):
            img_test_lists.append(os.path.join(path1, img_path))
            test_cats.append(cat_cnt)
        cat_cnt += 1

    train_data = img_list2numpy(img_lists, (28, 28))
    pickle.dump((train_data, cats), open('../datasets/omniglot/train_split.pkl', 'wb'))

    test_data = img_list2numpy(img_test_lists, (28, 28))
    pickle.dump((test_data, test_cats), open('../datasets/omniglot/test_split.pkl', 'wb'))

class OmniglotDataset:
    """
    Return a Omniglot dataset object
    """
    def __init__(self, is_train=True):
        data_path = '../datasets/omniglot/train_split.pkl' if is_train \
            else '../datasets/omniglot/test_split.pkl'

        data, labels = pickle.load(open(data_path, 'rb'))
        aug_data = np.zeros((data.shape[0] * 4, data.shape[1], data.shape[2], data.shape[3]))
        aug_labels = [0] * (len(labels) * 4)
        for i in range(len(data)):
            for dire in range(4):
                aug_data[i * 4 + dire] = np.rot90(data[i], k=dire, axes=(0, 1)).copy()
                aug_labels[i * 4 + dire] = labels[i] * 4 + dire

        self.data = aug_data
        self.labels = aug_labels

        trans_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(trans_list)
        self.is_train = is_train

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.labels[index]

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    # gen_train_test_pickle()
    train_dataset = OmniglotDataset(is_train=True)

