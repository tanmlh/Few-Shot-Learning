import os
import csv
import sys
import utils
import pickle
from PIL import Image
from tqdm import tqdm
import random
from torchvision import transforms 

sys.path.append('./')
import utils
from DataLoader import EpisodeLoader

def get_img_map(file_name):
    csv_file = csv.reader(open(file_name, 'r'))
    next(csv_file)
    img_list = []
    img_map = {}
    reverse_map = {}
    cls_map = {}
    cnt = 0
    for line in csv_file:
        img_list.append(line[0])
        if cls_map.get(line[1]) == None:
            cls_map[line[1]] = cnt
            reverse_map[cnt] = []
            cnt += 1
        reverse_map[cls_map[line[1]]].append(line[0])
        img_map[line[0]] = cls_map[line[1]]
    return img_list, img_map, reverse_map, cls_map

def gen_train_val_test_pickle(file_path, des_path):
    _ROOT_DIR = '../datasets/mini-imagenet/images'
    _IMG_SIZE = (84, 84)
    img_list, img_map, _, _ = get_img_map(file_path)
    img_pathes = [os.path.join(_ROOT_DIR, x) for x in img_list]
    data = utils.img_list2numpy(img_pathes, _IMG_SIZE)
    labels = [img_map[x] for x in img_list]
    pickle.dump((data, labels), open(des_path, 'wb'))

class MiniImageNetDataset:
    """
    Return a MiniImageNetDataset dataset object
    """
    def __init__(self, phase='train'):

        if phase == 'train' or phase == 'pretrain_train' or phase == 'pretrain_val':
            data_path = '../datasets/mini-imagenet/train.pkl'
            # data_path = '../datasets/mini-imagenet/miniImageNet_category_split_train_phase_train.pickle'
        elif phase == 'val':
            data_path = '../datasets/mini-imagenet/val.pkl'
        elif phase == 'test':
            data_path = '../datasets/mini-imagenet/test.pkl'
            # data_path = '../datasets/mini-imagenet/miniImageNet_category_split_test.pickle'
        elif phase == 'train_val':
            data_path = '../datasets/mini-imagenet/train_val.pkl'
        else:
            raise NotImplementedError

        print('loading %s file from %s ...' % (phase, data_path))


        self.data, self.labels = pickle.load(open(data_path, 'rb'))
        """
        pkl = pickle.load(open(data_path, 'rb'))
        self.data = pkl['data']
        self.labels = pkl['labels']
        """
        random.seed(2019)
        random_idxes = range(len(self.data))
        random.shuffle(random_idxes)
        num_train = int(len(self.data) * 0.9)
        if phase == 'pretrain_train':
            self.data = [self.data[i] for i in random_idxes[:num_train]]
            self.labels = [self.labels[i] for i in random_idxes[:num_train]]
        elif phase == 'pretrain_val':
            self.data = [self.data[i] for i in random_idxes[num_train:]]
            self.labels = [self.labels[i] for i in random_idxes[num_train:]]
        """
        elif phase == 'test':
            temp = [self.labels[i] - 80 for i in range(len(self.labels))]
            self.labels = temp
        """

        mean_pix = [x/255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize_trainsform = transforms.Normalize(mean=mean_pix, std=std_pix)
        if phase == 'train' or phase == 'pretrain_train':
            trans_list = [transforms.RandomCrop(84, padding=8),
                          transforms.RandomHorizontalFlip(),
                          transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                          transforms.ToTensor(),
                          normalize_trainsform]
        else:
            trans_list = [transforms.CenterCrop(84),
                          transforms.ToTensor(),
                          normalize_trainsform]

        self.transform = transforms.Compose(trans_list)

    def __getitem__(self, index):
        # return self.transform(Image.fromarray(self.data[index])), self.labels[index]
        return self.transform(self.data[index]), self.labels[index]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    train_file_path = '../datasets/mini-imagenet/train.csv'
    val_file_path = '../datasets/mini-imagenet/val.csv'
    test_file_path = '../datasets/mini-imagenet/test.csv'
    train_pkl_path = '../datasets/mini-imagenet/train.pkl'
    val_pkl_path = '../datasets/mini-imagenet/val.pkl'
    test_pkl_path = '../datasets/mini-imagenet/test.pkl'

    
    gen_train_val_test_pickle(train_file_path, train_pkl_path)
    gen_train_val_test_pickle(val_file_path, val_pkl_path)
    gen_train_val_test_pickle(test_file_path, test_pkl_path)

    # dataset = MiniImageNetDataset(phase='test')
    """
    train_pkl = pickle.load(open(train_pkl_path, 'rb'))
    val_pkl = pickle.load(open(val_pkl_path, 'rb'))
    temp = [x + 64 for x in val_pkl[1]]
    train_val_pkl = (train_pkl[0] + val_pkl[0], train_pkl[1] + temp)
    pickle.dump(train_val_pkl, open('../datasets/mini-imagenet/train_val.pkl', 'wb'))
    """

