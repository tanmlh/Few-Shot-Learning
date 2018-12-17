import sys
import os
import shutil
import random
import numpy as np
import itertools
import cv2

sys.path.append('../')

_ROOT_DIR = '../datasets/omniglot/data'

def rename_folder(root_dir='../datasets/omniglot/python/images_background', des_dir='../datasets/omniglot/train_data'):
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
                        os.mkdir(temp)
                    cv2.imwrite(os.path.join(temp, str(cnt)+'.png'), img_np)
                    cnt += 1

def test_one_shot(root_dir, net, ctx):
    avg_acc = 0
    for running in os.listdir(root_dir):
        cur_dir = os.path.join(root_dir, running)
        f = open(os.path.join(cur_dir, 'class_labels.txt'), 'r')
        cls_id = 1
        img_map = {}
        train_img_list = []
        test_img_list = []
        for line in f.readlines():
            line = line.strip('\n')
            test_file, train_file = line.split(' ')
            img_map[train_file] = cls_id
            img_map[test_file] = cls_id
            train_img_list.append(train_file)
            test_img_list.append(test_file)
            cls_id += 1
        train_loader = Function.get_class_loader(root_dir, train_img_list, img_map, (28, 28))
        test_loader = Function.get_class_loader(root_dir, test_img_list, img_map, (28, 28))
        ProtoNet.attach_labels(net, train_loader, ctx)
        label, acc = ProtoNet.predict(net, test_loader, ctx)
        avg_acc += acc
        print('%s: %4f' % (running, acc))
    print('avg: %4f' % (avg_acc / 20))

def get_episode_loader(nc, ns, nq):
    root_dir = '../datasets/omniglot/data'
    return Function.get_episode_lodaer_v2(root_dir, (28, 28), nc, ns, nq, num_workers=0)

def choices(seq, nc):
    return random.choices(seq, k=nc)
    lis = list(range(len(seq)))
    lis = itertools.repeat(seq)
    random.shuffle(lis)
    return [seq[x] for x in lis[:nc]]


def get_episode(nc, ns, nq, ctx_num=1, is_train=True):
    ctx_pathes = []

    for ctx_id in range(ctx_num):
        classes = choices(list(cls2img_path.keys()), nc)
        s_pathes = []
        q_pathes = []
        for cls in classes:
            samples_num = len(cls2img_path[cls])
            train_num = min(int(samples_num * 0.8), samples_num-1)
            test_num = samples_num - train_num

            list_a = cls2img_path[cls][:train_num] if is_train else cls2img_path[cls][train_num:]
            path_s = choices(list_a, ns)
            list_b = list(set(list_a).difference(set(path_s)))
            path_q = choices(list_b, nq)

            s_pathes += path_s
            q_pathes += path_q
        ctx_pathes += s_pathes + q_pathes
    dataset = Function.ClassDataset(root_dir, ctx_pathes, img_path2cls, (28, 28))
    data, cls_id = next(iter(DataLoader(dataset, nc * (ns+nq) * ctx_num, num_workers=10,
                                        pin_memory=True)))
    return data, cls_id



if __name__ == '__main__':
    rename_folder()

    """
    ctx_id = 0
    ctx=mx.gpu(ctx_id)

    root_dir = '../datasets/omniglot/data'
    img_map, reverse_map, img_list, cls_map, cls_reverse_map = Function.get_img_map(root_dir)
    nc = 10
    ns = 2
    nq = 2
    """
