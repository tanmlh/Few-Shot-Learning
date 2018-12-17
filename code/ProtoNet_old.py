import os
import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import nn
from mxnet import autograd, init
import numpy as np
import pickle
import math
import Omniglot

def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

# Blocks
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

class ResNet(nn.HybridBlock):
    """
    [64, 64, 128, 256, 512]
    """
    def __init__(self, layers=[2, 2, 2, 2], channels=[64, 64, 128, 256, 512], cls_num = 64,
                 thumbnail=True, **kwargs):
        super(ResNet, self).__init__(**kwargs)
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



def my_loss(data, nc, ns, nq):
    data = data.astype('float64')
    cls_data = nd.reshape(data[0:nc*ns], (nc, ns, -1))
    cls_center = nd.mean(cls_data, axis=1) + 1e-10
    data_center_dis = nd.norm(data[nc*ns:].expand_dims(axis=1) - cls_center.expand_dims(axis=0),
                              axis=2) ** 2


    weight = nd.zeros((nc*nq, nc), ctx=data.context, dtype='float64')
    for i in range(0, nc):
        weight[i*nq:i*nq+nq, i] = 1
    weight2 = 1 - weight

    temp1 = nd.log_softmax(-data_center_dis, axis=1)
    temp2 = nd.sum(temp1, axis=1)
    temp3 = nd.sum(-temp2)
    label = nd.argmin(data_center_dis, axis=1)
    return temp3 / (nc * nq), label

    loss1 = nd.sum(data_center_dis * weight)

    temp = nd.sum(nd.exp(- data_center_dis), axis=1)
    loss2 = nd.sum(nd.log(temp))


    if loss1 is np.nan or loss2 is np.nan:
        raise StopIteration

    return (loss1 + loss2) / (nc * nq), label

def cal_acc(label, nc, nq):
    correct_cnt = 0
    for i in range(nc):
        correct_cnt += nd.sum(label[i*nq:(i+1)*nq] == i).asscalar()
    return correct_cnt / (nc * nq)

def train_proto_net(gpu_id, train_loader, test_loader, net_name, net=None, nc=10, ns=3, nq=3,
                    epoch_num=100):
    ctx = mx.gpu(gpu_id)
    batch_size = nc * (ns + nq)

    # train_loader, test_loader = CelebA.get_episode_lodaer(nc, ns, nq)

    if net == None:
        net = ResNet()
        net.initialize(init=init.Xavier(), ctx=ctx)

    lr_scheduler = mx.lr_scheduler.FactorScheduler(2000, 0.001, 0.5)
    trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {'lr_scheduler':lr_scheduler})
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    for epoch in range(1, epoch_num+1):
        train_loss = 0
        train_acc = 0
        for data, cls_id in train_loader:
            data = data.copyto(ctx)

            with autograd.record():
                out = net(data)
                loss, label = my_loss(out, nc, ns, nq)

            loss.backward()
            # print(net(data)[0], loss)
            trainer.step(batch_size * len(train_loader))
            train_loss += loss.asscalar()
            train_acc += cal_acc(label, nc, nq)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        if test_loader is not None:
            test_loss = 0
            test_acc = 0
            for data, cls_id in test_loader:
                data = data.copyto(ctx)
                out = net(data)
                loss, label = my_loss(out, nc, ns, nq)

                test_loss += loss.asscalar()
                test_acc += cal_acc(label, nc, nq)
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
        if test_loader is None:
            print('epoch: %d train_loss: %.4f train_acc: %.4f' %
                  (epoch, train_loss, train_acc))

        else:
            print('epoch: %d train_loss: %.4f train_acc: %.4f test_loss %.4f test_acc %.4f' %
                  (epoch, train_loss, train_acc, test_loss, test_acc))

        if epoch % 50 == 0:
            if not os.path.exists('../model/' + net_name):
                os.mkdir('../model/' + net_name)
            net.save_parameters('../model/' + net_name + '/' + net_name + '_%04d' % (epoch))

    return net

def test_episode(net, test_episode_loader, ctx, nc, ns, nq, episode_num=20):
    test_losses = []
    test_accs = []
    for epoch in range(episode_num):
        test_loss = 0
        test_acc = 0
        for data, cls_id, img_id in test_episode_loader:
            data = data.copyto(ctx)
            out = net(data)
            loss, label = my_loss(out, nc, ns, nq)

            test_loss += loss.asscalar()
            test_acc += cal_acc(label, nc, nq)
        test_loss /= len(test_episode_loader)
        test_acc /= len(test_episode_loader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print('episode: %d test_loss %.4f test_acc %.4f' %
              (epoch, test_loss, test_acc))
    return test_accs, test_losses

def attach_labels(net, data_loader, ctx):
    cls_sum = {}
    cls_cnt = {}
    for data, cls_id, img_id in data_loader:
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
    for data, cls_id, img_id in data_loader:
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

if __name__ == '__main__':
    ctx = mx.gpu(0);
    nc = 30; ns = 2; nq = 20; batch_size = nc*(ns+nq)
    net_name = 'proto_net_omniglot'

    net = ResNet()
    net.initialize(init.Xavier(), ctx=ctx)

    train_loader, test_loader = Omniglot.get_episode_loader(nc, ns, nq)

    train_proto_net(0, train_loader, test_loader, net_name, net, nc, ns, nq)


    # net = ResNet()
    # net.load_parameters('../model/protoNet_0200', ctx=ctx)
