# ResNet Wide Version as in Qiao's Paper
import torch.nn as nn
import sys
import torch

sys.path.append('../')
from BaseModule import BaseModule
from Solver import Solver


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, conf):
        super(ResNet, self).__init__()
        block = BasicBlock
        layers = [4, 4, 4]
        num_classes = conf['num_classes']
        cfg = [160, 320, 640]
        in_channels = conf['in_channels'] if 'in_channels' in conf else 3

        self.inplanes = iChannels = int(cfg[0]/2)
        self.conv1 = nn.Conv2d(in_channels, iChannels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, cfg[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, cfg[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[2], layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(10, stride=1)
        # 512 * block.expansion
        self.fc = nn.Linear(cfg[2] * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # no FC here
        # return x

        class_prob = self.fc(x)

        res = {'feature': x, 'class_prob':class_prob}
        return res


class ResNetModule(BaseModule):
    def __init__(self, conf, *args):
        super(ResNetModule, self).__init__(*args)
        self.conf = conf
        self.net = {}
        self.net['feature'] = ResNet(conf['feature'])
        self.init_optimizer()
        if 'pre_trained' in conf['feature']:
            state = torch.load(open(conf['feature']['pre_trained'], 'rb'))
            print('loading pretrained network parameters...')
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

def create_model(conf):
    return ResNetModule(conf)
