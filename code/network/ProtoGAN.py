import torch
import torch.nn as nn
import sys

import FeatureModel
sys.path.append('../')
from Solver import Solver
from Omniglot import OmniglotDataset
from DataLoader import EpisodeLoader
from PrototypicalNetwork import PrototypicalNetwork

class GANSolver(Solver):
    def __init__(self, opt):
        super(GANSolver, self).__init__(opt)

    def process_batch(self, batch, is_train):
        """
        feed a batch into generation adversarial network
        """
        # set network mode
        if is_train:
            self.net.train()
        else:
            self.net.eval()

        self.set_tensors(batch, is_train)


        out_generation = self.net.forward_generation(self.tensors)

        cur_state = {}
        cur_state['loss'] = out_generation['loss'].item()
        cur_state['accuracy'] = out_generation['accuracy']

        # if this is training process, calculate discriminative loss and accuracy
        if is_train:

            out_discriminate = self.net.forward_discriminate(self.tensors)

            self.net.optimizer['feature'].zero_grad()
            (out_generation['loss'] + out_discriminate['reverse_loss']).backward(retain_graph=True)
            self.net.optimizer['feature'].step()

            self.net.optimizer['discriminate'].zero_grad()
            out_discriminate['loss'].backward()
            self.net.optimizer['discriminate'].step()

            cur_state['loss_discriminate'] = out_discriminate['loss'].item()
            cur_state['accuracy_discriminate'] = out_discriminate['accuracy']

        return cur_state

    def init_tensors(self):
        tensors = {}
        tensors['support_data'] = torch.FloatTensor()
        tensors['support_labels'] = torch.LongTensor()
        tensors['support_labels_one_hot'] = torch.FloatTensor()
        tensors['query_data'] = torch.FloatTensor()
        tensors['query_labels'] = torch.LongTensor()
        tensors['anchor_data'] = torch.FloatTensor()
        tensors['anchor_labels'] = torch.LongTensor()

        self.tensors = tensors

    def set_tensors(self, batch, is_train):
        if is_train:
            support_data, support_labels, query_data, query_labels, Knovel, anchor_data, anchor_labels = batch
        else:
            support_data, support_labels, query_data, query_labels, Knovel = batch

        self.tensors['support_data'].resize_(support_data.size()).copy_(support_data)
        self.tensors['support_labels'].resize_(support_labels.size()).copy_(support_labels)
        self.tensors['query_data'].resize_(query_data.size()).copy_(query_data)
        self.tensors['query_labels'].resize_(query_labels.size()).copy_(query_labels)
        one_hot_size = list(support_labels.size()) + [Knovel.size()[1],]
        support_labels_unsqueeze = self.tensors['support_labels'].unsqueeze(support_labels.dim())
        self.tensors['support_labels_one_hot'].resize_(one_hot_size). \
            fill_(0).scatter_(support_labels.dim(), support_labels_unsqueeze, 1)

        if is_train:
            self.tensors['anchor_data'].resize_(anchor_data.size()).copy_(anchor_data)
            self.tensors['anchor_labels'].resize_(anchor_labels.size()).copy_(anchor_labels)


class ProtoGAN(nn.Module):
    def __init__(self, opt):
        super(ProtoGAN, self).__init__()
        self.opt = opt
        img_size = opt['img_size']

        self.feature_net = FeatureModel.create_model(opt['feature_net_opt'])

        opt['discriminate_net_opt']['in_planes'] = self.feature_net.get_output_size(img_size)[1]
        self.discriminate_net = DiscriminateModel(opt['discriminate_net_opt'])

        self.proto_net = PrototypicalNetwork()

        self.init_optimizer()

    def forward_generation(self, tensors):
        support_data = tensors['support_data']
        support_labels = tensors['support_labels']
        query_data = tensors['query_data']
        query_labels = tensors['query_labels']
        support_labels_one_hot = tensors['support_labels_one_hot']

        batch_size, num_support, channel, height, weight = support_data.size()
        support_features = self.feature_net(support_data.view(batch_size*num_support, channel, height, weight))
        support_features = support_features.view(batch_size, num_support, -1)

        batch_size, num_query, channel, height, weight = query_data.size()
        query_features = self.feature_net(query_data.view(batch_size*num_query, channel, height, weight))
        query_features = query_features.view(batch_size, num_query, -1)

        out_proto = self.proto_net(support_features, support_labels, query_features, query_labels,
                                   support_labels_one_hot)
        out = out_proto

        return out

    def forward_discriminate(self, tensors):
        anchor_data = tensors['anchor_data']
        anchor_labels = tensors['anchor_labels']

        batch_size, num_anchors, channel, height, weight = anchor_data.size()
        anchor_data = anchor_data.view(batch_size*num_anchors, channel, height, weight)
        anchor_labels = anchor_labels.view(batch_size*num_anchors,)
        anchor_features = self.feature_net(anchor_data)

        out_discriminate = self.discriminate_net((anchor_features, anchor_labels))

        return out_discriminate

    def init_optimizer(self):
        feature_parameters = filter(lambda x: x.requires_grad, self.feature_net.parameters())
        discriminate_parameters = filter(lambda x: x.requires_grad, self.discriminate_net.parameters())

        feature_optimizer = torch.optim.SGD(feature_parameters,
                                            lr=0.001,momentum=0.9,
                                            weight_decay=5e-4,
                                            nesterov=True)

        discriminate_optimizer = torch.optim.SGD(discriminate_parameters,
                                                 lr=0.01,
                                                 momentum=0.9,
                                                 weight_decay=5e-4,
                                                 nesterov=True)

        self.optimizer = {'feature':feature_optimizer,
                          'discriminate':discriminate_optimizer}

    def adjust_lr(self, cur_epoch):
        LUT = self.opt['LUT_lr']
        if LUT is None:
            LUT = [(2, 0.1), (10, 0.006), (20, 0.0012), (40, 0.00024)]
        for epoch, lr in LUT:
            if cur_epoch < epoch:
                for param_group in self.optimizer['feature'].param_groups:
                    param_group['lr'] = lr
                break


class DiscriminateModel(nn.Module):

    def __init__(self, opt):
        super(DiscriminateModel, self).__init__()

        in_features = opt['in_planes']
        out_features = opt['out_planes']

        num_features = [in_features,] + out_features
        self.linear_layers = nn.Sequential()
        for i in range(len(num_features)-1):
            self.linear_layers.add_module('layer'+str(i+1), nn.Linear(num_features[i], num_features[i+1]))

        self.cls_layer = nn.Linear(num_features[-1], 2)
        self.loss_func = nn.CrossEntropyLoss()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, batch):
        x, labels = batch
        x_out = self.linear_layers(x)
        x_out = self.cls_layer(x_out)

        loss = self.loss_func(x_out, labels)
        reverse_loss = self.loss_func(x_out, 1-labels)
        accuracy = 1.0 * (x_out.argmax(dim=1) == labels).sum().item() / labels.size()[0]
        res = {'loss': loss, 'accuracy' : accuracy, 'reverse_loss' : reverse_loss}
        return res

def create_model(opt):
    return ProtoGAN(opt)


