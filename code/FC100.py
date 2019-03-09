import os
import csv
import sys
import utils
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
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

class FC100Dataset:
    """
    Return a MiniImageNetDataset dataset object
    """
    def __init__(self, phase='train'):

        if phase == 'train' or phase == 'pretrain_train' or phase == 'pretrain_val':
            data_path = '../datasets/FC100/few-shot-train.npz'
        elif phase == 'val':
            data_path = '../datasets/FC100/few-shot-val.npz'
        elif phase == 'test':
            data_path = '../datasets/FC100/few-shot-test.npz'
        else:
            raise NotImplementedError

        print('loading %s file from %s ...' % (phase, data_path))

        temp  = np.load(data_path)
        self.data, self.labels = temp['features'], temp['targets']
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
            trans_list = [transforms.RandomCrop(32, padding=2),
                          transforms.RandomHorizontalFlip(),
                          transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                          transforms.ToTensor(),
                          normalize_trainsform]
        else:
            trans_list = [transforms.CenterCrop(32),
                          transforms.ToTensor(),
                          normalize_trainsform]

        self.transform = transforms.Compose(trans_list)

    def __getitem__(self, index):
        # return self.transform(Image.fromarray(self.data[index])), self.labels[index]
        return self.transform(Image.fromarray(self.data[index])), self.labels[index]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    dataset = FC100Dataset(phase='train')
    temp = dataset[0]
    """
    train_pkl = pickle.load(open(train_pkl_path, 'rb'))
    val_pkl = pickle.load(open(val_pkl_path, 'rb'))
    temp = [x + 64 for x in val_pkl[1]]
    train_val_pkl = (train_pkl[0] + val_pkl[0], train_pkl[1] + temp)
    pickle.dump(train_val_pkl, open('../datasets/mini-imagenet/train_val.pkl', 'wb'))
    """

