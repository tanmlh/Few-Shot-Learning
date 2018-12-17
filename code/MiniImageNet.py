import os
import csv
from mxnet.gluon.data import DataLoader, Dataset
from . import ProtoNet
from mxnet import autograd, init
import mxnet as mx
import sys

sys.path.append('../')
from common import Function


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

def get_loader(csv_file_name, img_size, nc, ns, nq):
    img_list, img_map, reverse_map, cls_map = get_img_map(csv_file_name)

    loader_list = {}
    for cls in reverse_map.keys():
        dataset = Function.ClassDataset('../mini-imagenet/images', reverse_map[cls],
                                                     img_map, img_size)
        loader_list[cls] = DataLoader(dataset, batch_size=ns+nq, num_workers=0, shuffle=True,
                                      last_batch='rollover')
    return Function.EpisodeLoader(loader_list, nc, ns, nq)

if __name__ == '__main__':
    img_size = (84, 84)
    nc = 30
    ns = 1
    nq = 15
    ctx = mx.gpu(2)
    ctx_id=2

    train_loader = get_loader('../mini-imagenet/train.csv', img_size, nc, ns, nq)
    val_loader = get_loader('../mini-imagenet/test.csv', img_size, nc, ns, nq)
    net = ProtoNet.FourBlocks()
    # net.load_parameters('../model/ProtoNet_miniImageNet_0650', ctx=ctx)
    # test_loader = get_loader('../mini-imagenet/test.csv', img_size, 5, 5, 20)
    # test_accs, test_losses = ProtoNet.test_episode(net, test_loader, ctx, 5, 5, 20)
    net.initialize(init=init.Xavier(), ctx=ctx)
    net = ProtoNet.train_proto_net(ctx_id, train_loader, val_loader, 'ProtoNet_miniImageNet',
                                   nc=nc, ns=ns, nq=nq, epoch_num=1000)

