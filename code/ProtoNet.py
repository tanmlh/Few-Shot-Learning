import sys
import os
import numpy as np
import random
import math

import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import nn
from mxnet import init, autograd, gluon

sys.path.append('../')
from common import Function
import Omniglot

def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

class FourBlocks(nn.HybridSequential):
    def __init__(self, **args):
        super(FourBlocks, self).__init__(prefix='cnn_')
        num_layers = [0, 64, 64, 64, 64]
        with self.name_scope():
            for i in range(4):
                self.add(_conv3x3(num_layers[i+1], 1, num_layers[i]))
                self.add(nn.BatchNorm())
                self.add(nn.Activation('relu'))
                self.add(nn.MaxPool2D())
            self.add(nn.Flatten())
            self.add(nn.Dense(64, activation='relu'))

class ResBlock(nn.HybridBlock):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        self.body.add(nn.BatchNorm())
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)
        x = F.Activation(residual+x, act_type='relu')
        return x
class FourBlocks(nn.HybridBlock):
    """
    [64, 64, 128, 256, 512]
    """
    def __init__(self, layers=[2, 2, 2, 2], channels=[64, 64, 128, 256, 512], cls_num = 64,
                 thumbnail=True, **kwargs):
        super(FourBlocks, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        self.features = nn.HybridSequential(prefix='')

        if not thumbnail:
            self.features.add(_conv3x3(channels[0], 1, 0))
        else:
            self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(3, 2, 1))

        self.channels = channels
        self.layers = layers
        self.output = nn.Dense(cls_num)

        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            self.features.add(self._make_layer(num_layer, channels[i+1],
                                               stride, i+1, in_channels=channels[i]))

    def _make_layer(self, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(ResBlock(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix=''))
            for _ in range(layers-1):
                layer.add(ResBlock(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        feature = self.features(x)
        feature = self.output(feature)
        return feature


def attach_labels(net, data_loader, ctx):
    cls_sum = {}
    cls_cnt = {}
    for data, cls_id in data_loader:
        cls_id = cls_id.asscalar()
        data = data.as_in_context(ctx)
        if cls_sum.get(cls_id) is None:
            cls_sum[cls_id] = net(data)
            cls_cnt[cls_id] = 1
        else:
            cls_sum[cls_id] += net(data)
            cls_cnt[cls_id] += 1
    for key in cls_sum.keys():
        cls_sum[key] /= cls_cnt[key]
    net.cls_center = cls_sum

def predict(net, data_loader, ctx):
    label = []
    acc = 0
    for data, cls_id in data_loader:
        data = data.as_in_context(ctx)
        out = net(data)
        min_dis = math.inf
        p_key = None
        for key in net.cls_center:
            cur_dis = nd.norm(net.cls_center[key] - out)
            if cur_dis.asscalar() < min_dis:
                min_dis = cur_dis.asscalar()
                p_key = key
        if p_key == cls_id.asscalar():
            acc += 1
        label.append(p_key)

    return label, acc / len(label)

def proto_loss(embedding, nc, ns, nq):
    embedding = embedding.astype('float64');
    cls_data = nd.reshape(embedding[0:nc*ns], (nc, ns, -1)); cls_data.attach_grad()
    cls_center = nd.mean(cls_data, axis=1);
    data_center_dis = nd.norm(embedding[nc*ns:].expand_dims(axis=1) - cls_center.expand_dims(axis=0),
                              axis=2) ** 2

    # print(nd.max(data_center_dis).asscalar())


    weight = nd.zeros((nc*nq, nc), ctx=embedding.context, dtype='float64')
    pick_vec = nd.zeros((nc*nq), ctx=embedding.context)
    for i in range(0, nc):
        weight[i*nq:i*nq+nq, i] = 1
        pick_vec[i*nq:i*nq+nq] = i
    """
    temp = nd.SoftmaxOutput(-data_center_dis, label)
    temp = nd.log(temp) * weight
    temp = nd.sum(-temp, axis=1)
    predict = nd.argmin(data_center_dis, axis=1)
    return -temp * nd.log(temp), predict
    """

    temp1 = nd.log_softmax(-data_center_dis, axis=1);
    temp2 = nd.pick(temp1, index=pick_vec, axis=1);
    temp3 = nd.sum(-temp2);
    label = nd.argmin(data_center_dis, axis=1)
    return temp3 / (nc * nq), label

def cal_acc(label, nc, nq):
    correct_cnt = 0
    for i in range(nc):
        correct_cnt += int(np.sum(label[i*nq:(i+1)*nq] == i))
    return correct_cnt / (nc * nq)

def forward_batch(net, data, ctx, nc, ns, nq, is_train=True):

    data = gluon.utils.split_and_load(data, ctx)
    loss = []
    label = []

    if is_train:
        with autograd.record():
            for X in data:
                out = net(X)
                _loss, _label = proto_loss(out, nc, ns, nq)
                loss.append(_loss)
                label.append(_label)
    else:
        for X in data:
            out = net(X)
            _loss, _label = proto_loss(out, nc, ns, nq)
            loss.append(_loss)
            label.append(_label)

    out_label = np.concatenate([_label.asnumpy() for _label in label])

    out_loss = 0
    for l in loss:
        if is_train:
            l.backward()
        out_loss += l.asscalar()

    return out_loss, out_label

class EpisodeProvider:
    def __init__(self, loader):
        self.loader = loader
        self.ite = iter(loader)

    def next(self):
        try:
            data, cls_id = next(self.ite)
        except StopIteration:
            self.ite = iter(train_loader)
            data, cls_id = next(self.ite)
        return data, cls_id

def train_proto_net(ctx_ids, train_loader, val_loader, net_name, nc, ns, nq, episode_num=2000):
    ctx = [mx.gpu(x) for x in ctx_ids]; ctx_num = len(ctx)
    batch_size = nc*(ns+nq)
    net_name = 'proto_net_omniglot'

    net = FourBlocks(prefix='cnn_')
    net.initialize(init.Xavier(), ctx=ctx)

    iter_train = EpisodeProvider(train_loader)
    iter_test = EpisodeProvider(test_loader)

    lr_scheduler = mx.lr_scheduler.FactorScheduler(1000, 0.001, 0.5)
    trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {'lr_scheduler':lr_scheduler})
    # trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':0.01})
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    # weight = net.collect_params()['cnn_dense0_weight']
    max_test_acc = 0
    for ite_id in range(1, episode_num+1):

        data, cls_id = iter_train.next()
        train_loss, train_label = forward_batch(net, data, ctx, nc, ns, nq)
        trainer.step(batch_size)
        # print(net.collect_params()['cnn_dense0_weight'].data(ctx=ctx[0]))

        train_losses.append(train_loss)
        train_accs.append(cal_acc(train_label, nc, nq))

        test_data, test_cls_id = iter_test.next()
        test_loss, test_label = forward_batch(net, test_data, ctx, nc, ns, nq, False)

        test_losses.append(test_loss)
        test_accs.append(cal_acc(test_label, nc, nq))

        if test_accs[-1] >= max_test_acc:
            max_test_acc = test_accs[-1]
            if not os.path.exists('../model/' + net_name):
                os.mkdir('../model/' + net_name)
            net.save_parameters('../model/' + net_name + '/model_best.params')

        if ite_id % 100 == 0:
            print('episode: %d train_loss: %.4f train_acc: %.4f test_loss %.4f test_acc %.4f' %
                  (ite_id, train_loss, train_accs[-1], test_loss, test_accs[-1]))

        if ite_id % 1000 == 0:
            if not os.path.exists('../model/' + net_name):
                os.mkdir('../model/' + net_name)
            net.save_parameters('../model/' + net_name + '/model_%04d.params' % (ite_id))



if __name__ == '__main__':
    ctx = [mx.gpu(0)]; ctx_num = len(ctx)
    model_save_path = '../model/proto_net/model'
    nc = 30; ns = 2; nq = 20; batch_size = nc*(ns+nq)
    net_name = 'proto_net_omniglot'

    net = FourBlocks(prefix='cnn_')
    net.initialize(init.Xavier(), ctx=ctx)

    train_loader, test_loader = Omniglot.get_episode_loader(nc, ns, nq)
    iter_train = EpisodeProvider(train_loader)
    iter_test = EpisodeProvider(test_loader)

    train_proto_net([0], train_loader, test_loader, net_name, nc, ns, nq)

    # net.hybridize()
    # test_data = nd.ones((2, 3, 64, 64))
    # test_data = gluon.utils.split_and_load(test_data, ctx)
    # res = [net(data) for data in test_data]
    # net.export(model_save_path, 0)
    # Function.visualize_symbol(model_save_path, 0)
