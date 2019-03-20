import sys
import torch
from torch import nn

sys.path.append('./network')

import ConvNet
import WideResNet
import ProtoHead
from BaseModule import BaseModule
sys.path.append('../')
from Solver import Solver

def get_solver(conf):
    return ProtoSolver(conf)

class ProtoSolver(Solver):
    def __init__(self, *args):
        super(ProtoSolver, self).__init__(*args)

    def init_tensors(self):
        tensors = {}
        tensors['support_data'] = torch.FloatTensor()
        tensors['support_labels'] = torch.LongTensor()
        tensors['support_labels_one_hot'] = torch.FloatTensor()
        tensors['query_data'] = torch.FloatTensor()
        tensors['query_labels'] = torch.LongTensor()
        tensors['query_labels_real'] = torch.LongTensor()

        self.tensors = tensors

    def set_tensors(self, batch):
        support_data, support_labels, query_data, query_labels, query_labels_real = batch
        num_way = query_labels.max().item()+1

        self.tensors['support_data'].resize_(support_data.size()).copy_(support_data)
        self.tensors['support_labels'].resize_(support_labels.size()).copy_(support_labels)
        self.tensors['query_data'].resize_(query_data.size()).copy_(query_data)
        self.tensors['query_labels'].resize_(query_labels.size()).copy_(query_labels)
        self.tensors['query_labels_real'].resize_(query_labels_real.size()).copy_(query_labels_real)
        one_hot_size = list(support_labels.size()) + [num_way,]
        support_labels_unsqueeze = self.tensors['support_labels'].unsqueeze(support_labels.dim())
        self.tensors['support_labels_one_hot'].resize_(one_hot_size). \
            fill_(0).scatter_(support_labels.dim(), support_labels_unsqueeze, 1)

    def process_batch(self, batch, is_train=True):
        """
        * Feed a batch into the network
        * calculate the loss
        * apply optimization step
        * return the state
        """
        if is_train:
            self.net.train()
            self.net.zero_grad()
        else:
            self.net.eval()

        self.set_tensors(batch)
        out = self.net(self.tensors)

        if is_train:
            out['loss'].backward()
            self.net.step()

        cur_state = {}
        cur_state['loss'] = out['loss'].item()
        cur_state['accuracy_classification'] = out['accuracy_classification']
        cur_state['accuracy'] = out['accuracy_proto']

        return cur_state

    def generate_case(self, batch):
        support_data, support_labels, query_data, query_labels, Knovel = batch
        state = self.process_batch(batch, is_train=False)
        case = []
        for i in range(support_data.size()[0]):
            cur_case = {}
            cur_case['data'] = torch.cat([support_data[i], query_data[i]])
            cur_case['pred_labels'] = state['pred_labels'][i]
            cur_case['gt_labels'] = state['gt_labels'][i]
            case.append(cur_case)

        return case

    def print_state(self, state, epoch, phase):
        print('%s %d   --> acc: %f | acc_class %f' % (phase,
                                                      epoch,
                                                      state['accuracy'],
                                                      state['accuracy_classification']))


class ProtoNetModule(BaseModule):
    def __init__(self, conf, *args):
        super(ProtoNetModule, self).__init__(*args)
        self.conf = conf
        self.net = {}
        if conf['feature']['net_name'] == 'WideResNet':
            self.net['feature'] = WideResNet.create_model(conf)
        else:
            self.net['feature'] = ConvNet.create_model(conf['feature'])

        self.net['head'] = ProtoHead.create_model()
        self.init_optimizer()

    def forward(self, tensors):
        support_data = tensors['support_data']
        support_labels = tensors['support_labels']
        query_data = tensors['query_data']
        query_labels = tensors['query_labels']
        support_labels_one_hot = tensors['support_labels_one_hot']
        query_labels_real = tensors['query_labels_real']

        batch_size, num_support, channel, height, weight = support_data.size()
        out_support = self.net['feature'](support_data.view(batch_size*num_support, channel, height, weight))
        support_features = out_support['feature']
        support_features = support_features.view(batch_size, num_support, -1)

        batch_size, num_query, channel, height, weight = query_data.size()
        out_query = self.net['feature'](query_data.view(batch_size*num_query, channel, height, weight),
                                        query_labels_real.view(batch_size*num_query))
        query_features = out_query['feature']
        query_features = query_features.view(batch_size, num_query, -1)

        out_proto = self.net['head'](support_features, support_labels, query_features, query_labels,
                                   support_labels_one_hot)

        out = {}
        out['loss_proto'] = out_proto['loss']
        out['accuracy_proto'] = out_proto['accuracy']
        out['loss_classification'] = out_query['loss']
        out['accuracy_classification'] = out_query['accuracy']

        ratio = self.conf['ratio']
        out['loss'] = 0
        if ratio[0] != 0:
            out['loss'] += out['loss_proto']
        if ratio[1] != 0:
            out['loss'] += out['loss_classification']

        return out
    """
    def adjust_lr(self, cur_epoch):
        LUT = self.conf['LUT_lr']
        if LUT is None:
            LUT = [(2, 0.1), (10, 0.006), (20, 0.0012), (40, 0.00024)]
        for epoch, lr in LUT:
            if cur_epoch < epoch:
                for param_group in self.optimizer['feature'].param_groups:
                    param_group['lr'] = lr
                break
    """

def create_model(conf):
    return ProtoNetModule(conf)
