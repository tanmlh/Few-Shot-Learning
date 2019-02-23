import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('../')
from BaseModule import BaseModule
from Solver import Solver

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, userelu=True):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
            kernel_size=3, stride=1, padding=1, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))

        if userelu:
            self.layers.add_module('ReLU', nn.ReLU(inplace=True))

        self.layers.add_module(
            'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.layers(x)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                 padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv0 = conv1x1(inplanes, planes, stride)

        self.conv1 = conv3x3(inplanes, planes, 1)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.9)

    def forward(self, x):
        identity = self.conv0(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)
        # out = self.relu(out)

        out += identity
        out = self.max_pool(out)
        # out = self.dropout(out)

        return out


class ConvNet(nn.Module):
    def __init__(self, opt):
        super(ConvNet, self).__init__()
        self.in_planes  = opt['in_planes']
        self.out_planes = opt['out_planes']
        self.num_stages = opt['num_stages']
        self.num_classes = opt['num_classes']


        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert(type(self.out_planes)==list and len(self.out_planes)==self.num_stages)

        num_planes = [self.in_planes,] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else True

        conv_blocks = []
        for i in range(self.num_stages):
            if i == (self.num_stages-1):
                conv_blocks.append(
                    BasicBlock(num_planes[i], num_planes[i+1]))
            else:
                conv_blocks.append(
                    BasicBlock(num_planes[i], num_planes[i+1]))
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_planes[-1] * 6 * 6, self.num_classes)
        self.relu = nn.ReLU()
        self.initialization()

    def initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_blocks(x)
        feature = x.view(x.size(0), -1)
        class_prob = self.fc(feature)
        # class_prob = self.relu(class_prob)

        out = {'feature':feature, 'class_prob':class_prob}
        return out

def get_solver(conf):
    return ConvNetSolver(conf)

class ConvNetSolver(Solver):

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

def create_model(opt=None):
    if opt is None:
        opt = {}
        opt['userelu'] = True; opt['in_planes'] = opt['img_size'][0]
        opt['out_planes'] = [64, 64, 128, 128]; opt['num_stages'] = 4
    return ConvNetModule(opt)

class ConvNetModule(BaseModule):
    def __init__(self, opt, *args):
        super(ConvNetModule, self).__init__(*args)
        self.opt = opt
        self.net = {}
        self.net['feature'] = ConvNet(opt)
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
