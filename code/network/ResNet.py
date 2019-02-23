import sys
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

sys.path.append('../')
from BaseModule import BaseModule
from Solver import Solver

class ResBlock(nn.Module):
    def __init__(self, nFin, nFout):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential()
        self.conv_block.add_module('BNorm1', nn.BatchNorm2d(nFin))
        self.conv_block.add_module('LRelu1', nn.LeakyReLU(0.1))
        self.conv_block.add_module('ConvL1',
            nn.Conv2d(nFin,  nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm2', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu2', nn.LeakyReLU(0.1))
        self.conv_block.add_module('ConvL2',
            nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))
        self.conv_block.add_module('BNorm3', nn.BatchNorm2d(nFout))
        self.conv_block.add_module('LRelu3', nn.LeakyReLU(0.1))
        self.conv_block.add_module('ConvL3',
            nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))

        self.skip_layer = nn.Conv2d(nFin, nFout, kernel_size=1, stride=1)
        ## 
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

    def forward(self, x):
        x = self.skip_layer(x) + self.conv_block(x)
        return x


class ResNetLike(nn.Module):
    def __init__(self, opt):
        super(ResNetLike, self).__init__()
        self.in_planes = opt['in_planes']
        self.out_planes = [64, 96, 128, 256]
        self.num_stages = 4

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert(type(self.out_planes)==list)
        assert(len(self.out_planes)==self.num_stages)
        num_planes = [self.out_planes[0],] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else False
        dropout = opt['dropout'] if ('dropout' in opt) else 0
        num_classes = opt['num_classes']

        self.conv_0 = nn.Conv2d(self.in_planes, num_planes[0], kernel_size=3, padding=1)
        self.res_block_0 = ResBlock(num_planes[0], num_planes[1])
        self.res_block_1 = ResBlock(num_planes[1], num_planes[2])
        self.res_block_2 = ResBlock(num_planes[2], num_planes[3])
        self.res_block_3 = ResBlock(num_planes[3], num_planes[4])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.bn_1 = nn.BatchNorm2d(num_planes[4])
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(num_planes[4], 384, kernel_size=1)
        self.dropout = nn.Dropout(p=0.9)
        self.bn_2 = nn.BatchNorm2d(384)
        self.conv_2 = nn.Conv2d(384, 512, kernel_size=1)
        self.bn_3 = nn.BatchNorm2d(512)

        self.feat_extractor = nn.Sequential()
        self.feat_extractor.add_module('ConvL0',
            nn.Conv2d(self.in_planes, num_planes[0], kernel_size=3, padding=1))
        for i in range(self.num_stages):
            self.feat_extractor.add_module('ResBlock'+str(i),
                ResBlock(num_planes[i], num_planes[i+1]))
            self.feat_extractor.add_module('MaxPool'+str(i),
                nn.MaxPool2d(kernel_size=2,stride=2,padding=0))

        self.feat_extractor.add_module('AvgPool', nn.AdaptiveAvgPool2d(1))
        self.feat_extractor.add_module('BNormF1',
            nn.BatchNorm2d(num_planes[-1]))
        self.feat_extractor.add_module('ReluF1', nn.ReLU(inplace=True))
        self.feat_extractor.add_module('ConvLF1',
            nn.Conv2d(num_planes[-1], 384, kernel_size=1))
        if dropout>0.0:
            self.feat_extractor.add_module('DropoutF1',
                nn.Dropout(p=dropout, inplace=False))

        self.feat_extractor.add_module('BNormF2', nn.BatchNorm2d(384))
        self.feat_extractor.add_module('ReluF2', nn.ReLU(inplace=True))
        self.feat_extractor.add_module('ConvLF2',
            nn.Conv2d(384, 512, kernel_size=1))
        self.feat_extractor.add_module('BNormF3', nn.BatchNorm2d(512))

        self.fc = nn.Linear(512, num_classes)

        if dropout>0.0:
            self.feat_extractor.add_module('DropoutF2',
                nn.Dropout(p=dropout, inplace=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        feature = self.feat_extractor(x)
        feature = feature.view(feature.size(0), -1)
        class_prob = self.fc(feature)
        out = {'feature':feature, 'class_prob':class_prob}
        """
        x = self.conv_0(x)
        x = self.res_block_0(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.avg_pool(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.dropout(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.conv_2(x)
        feature = self.bn_3(x)
        class_prob = self.fc(feature.view(feature.size(0), -1))
        out = {'feature':feature, 'class_prob':class_prob}
        """

        return out

def get_solver(conf):
    return ResNetSolver(conf)

class ResNetSolver(Solver):

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
    return ResNetModule(opt)

class ResNetModule(BaseModule):
    def __init__(self, opt, *args):
        super(ResNetModule, self).__init__(*args)
        self.opt = opt
        self.net = {}
        self.net['feature'] = ResNetLike(opt)
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
