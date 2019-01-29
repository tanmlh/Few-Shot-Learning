import os
import csv
import sys
import utils
import pickle
from PIL import Image
from tqdm import tqdm

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
    cnt = 1
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

        if phase == 'train':
            data_path = '../datasets/mini-imagenet/train.pkl'
        elif phase == 'val':
            data_path = '../datasets/mini-imagenet/val.pkl'
        elif phase == 'test':
            data_path = '../datasets/mini-imagenet/test.pkl'
        else:
            raise NotImplementedError


        self.data, self.labels = pickle.load(open(data_path, 'rb'))
        mean_pix = [x/255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize_trainsform = transforms.Normalize(mean=mean_pix, std=std_pix)
        if phase == 'train':
            trans_list = [transforms.RandomCrop(84, padding=8),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          normalize_trainsform]
        else:
            trans_list = [transforms.CenterCrop(84),
                          transforms.ToTensor(),
                          normalize_trainsform]

        self.transform = transforms.Compose(trans_list)

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.labels[index]
        # return transforms.ToTensor()(self.data[index]), self.labels[index]

    def __len__(self):
        return self.data.shape[0]

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

