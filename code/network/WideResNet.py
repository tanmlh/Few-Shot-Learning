import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
from BaseModule import BaseModule
from Solver import Solver

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, opt):
        super(WideResNet, self).__init__()

        depth = opt['depth'] if 'depth' in opt else 22
        num_classes = opt['num_classes']
        widen_factor = 8
        dropRate = opt['drop_rate'] if 'drop_rate' in opt else 0.0

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3] * 2 * 2, num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)

        feature = F.adaptive_avg_pool2d(out, 1).view(x.size(0), -1)
        # feature = F.avg_pool2d(out, 8).view(x.size(0), -1)

        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(x.size(0), -1)

        # feature = out

        class_prob = self.fc(out)

        res = {'feature': feature, 'class_prob':class_prob}
        return res

def get_solver(conf):
    return WideResNetSolver(conf)

class WideResNetSolver(Solver):

    def init_tensors(self):
        """
        Define all the tensors that will be used in network computation.
        """
        tensors = {}
        tensors['data'] = torch.FloatTensor()
        tensors['labels'] = torch.LongTensor()
        self.tensors = tensors

    def set_tensors(self, batch):
        """
        Set all the tensors that will be used in network computation.
        """
        data, labels = batch

        batch_size = data.size(0)

        self.tensors['data'].resize_(data.size()).copy_(data)
        self.tensors['labels'].resize_(labels.size()).copy_(labels)

    def process_batch(self, batch, is_train):
        """
        Process a batch of data
        """
        self.set_tensors(batch)
        if is_train:
            self.net.train()
            self.net.zero_grad()
        else:
            self.net.eval()

        self.set_tensors(batch)
        out = self.net(self.tensors['data'], self.tensors['labels'])

        if is_train:
            out['loss'].backward()
            self.net.step()

        cur_state = {}
        cur_state['loss'] = out['loss'].item()
        cur_state['accuracy'] = out['accuracy']
        return cur_state

    def print_state(self, state, epoch, is_train):
        if is_train:
            print('Training   epoch %d   --> accuracy: %f' % (epoch,
                                                              state['accuracy']))

        else:
            print('Evaluating epoch %d --> accuracy: %f' % (epoch,
                                                            state['accuracy']))

def create_model(opt):
    return WideResNetModule(opt)

class WideResNetModule(BaseModule):
    def __init__(self, opt, *args):
        super(WideResNetModule, self).__init__(*args)
        self.opt = opt
        self.net = {}
        self.net['feature'] = WideResNet(opt)
        self.init_optimizer()
        if 'pre_trained' in opt:
            state = torch.load(open(opt['pre_trained'], 'rb'))
            self.load_net_state(state)

    def forward(self, data, labels=None):

        batch_size = data.size(0)

        out = self.net['feature'](data)
        class_prob = out['class_prob']
        feature = out['feature']

        loss_func = nn.CrossEntropyLoss()

        loss = 0
        accuracy = 0
        if labels is not None:
            loss = loss_func(class_prob, labels)
            accuracy = (class_prob.argmax(dim=1) == labels).sum().item() / (1.0 * batch_size)

        out = {'loss':loss, 'accuracy':accuracy, 'feature':feature}

        return out

    def parameters(self):
        return self.net['feature'].parameters()

    def state_dict(self):
        return self.net['feature'].state_dict()

    def load_state_dict(self, state):
        self.net['feature'].load_state_dict(state)
