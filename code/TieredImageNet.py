import os
import csv
import sys
import utils
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import random
from torchvision import transforms 

sys.path.append('./')
import utils
from DataLoader import EpisodeLoader

def map_label(labels):
    label_map = {}
    mapped_labels = []
    cnt = 0
    for label in labels:
        if label not in label_map:
            label_map[label] = cnt
            cnt += 1
        mapped_labels.append(label_map[label])
    return mapped_labels

def decompress(in_path, out_path):
    with open(in_path, 'rb') as f:
        array = pickle.load(f)
        images = np.zeros([len(array), 84, 84, 3], dtype=np.uint8)
        for ii, item in tqdm(enumerate(array), desc='decompress'):
            img = cv2.imdecode(item, 1)
            images[ii] = img
        np.savez(out_path, images=images)

class TieredImageNetDataset:
    """
    Return a TieredImageNetDataset dataset object
    """
    def __init__(self, phase='train'):

        if phase == 'train' or phase == 'pretrain_train' or phase == 'pretrain_val':
            data_path = '../datasets/tiered-imagenet/train.npz'
            label_path = '../datasets/tiered-imagenet/train_labels.pkl'
        elif phase == 'val':
            data_path = '../datasets/tiered-imagenet/val.npz'
            label_path = '../datasets/tiered-imagenet/val_labels.pkl'
        elif phase == 'test':
            data_path = '../datasets/tiered-imagenet/test.npz'
            label_path = '../datasets/tiered-imagenet/test_labels.pkl'
        else:
            raise NotImplementedError

        print('loading %s file from %s ...' % (phase, data_path))

        self.data  = np.load(data_path)['images']
        self.labels = np.load(label_path)['label_specific']
        """
        pkl = pickle.load(open(data_path, 'rb'))
        self.data = pkl['data']
        self.labels = pkl['labels']
        """
        random.seed(0)
        random_idxes = range(len(self.data))
        random.shuffle(random_idxes)
        num_train = int(len(self.data) * 0.8)
        if phase == 'pretrain_train':
            self.data = [self.data[i] for i in random_idxes[:num_train]]
            self.labels = [self.labels[i] for i in random_idxes[:num_train]]
        elif phase == 'pretrain_val':
            self.data = [self.data[i] for i in random_idxes[num_train:]]
            self.labels = [self.labels[i] for i in random_idxes[num_train:]]

        self.labels = map_label(self.labels)

        mean_pix = [x/255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize_trainsform = transforms.Normalize(mean=mean_pix, std=std_pix)
        if phase == 'train' or phase == 'pretrain_train':
            trans_list = [transforms.RandomCrop(84, padding=8),
                          transforms.RandomHorizontalFlip(),
                          transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                          transforms.ToTensor(),
                          normalize_trainsform]
        else:
            trans_list = [transforms.CenterCrop(84),
                          transforms.ToTensor(),
                          normalize_trainsform]

        self.transform = transforms.Compose(trans_list)

    def __getitem__(self, index):
        # return self.transform(Image.fromarray(self.data[index])), self.labels[index]
        return self.transform(Image.fromarray(self.data[index])), self.labels[index]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    train_imgs_path = '../datasets/tiered-imagenet/train_images_png.pkl'
    val_imgs_path = '../datasets/tiered-imagenet/val_images_png.pkl'
    test_imgs_path = '../datasets/tiered-imagenet/test_images_png.pkl'

    out_train_imgs_path = '../datasets/tiered-imagenet/train.pkl'
    out_val_imgs_path = '../datasets/tiered-imagenet/val.pkl'
    out_test_imgs_path = '../datasets/tiered-imagenet/test.pkl'

    decompress(train_imgs_path, out_train_imgs_path)
    decompress(val_imgs_path, out_val_imgs_path)
    decompress(test_imgs_path, out_test_imgs_path)

