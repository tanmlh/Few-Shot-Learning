import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

sys.path.append('../network')
sys.path.append('./')
import gnn_iclr
from Solver import Solver

def get_solver(conf):
    return GraphConvSolver(conf)

def create_model(opt):
    return GraphConvNet(opt)

class GraphConvSolver(Solver):
    """
    Solver for GNN
    """
    def init_tensors(self):
        tensors = {}
        tensors['support_data'] = torch.FloatTensor()
        tensors['support_labels'] = torch.LongTensor()
        tensors['support_labels_one_hot'] = torch.FloatTensor()
        tensors['query_data'] = torch.FloatTensor()
        tensors['query_labels'] = torch.LongTensor()
        tensors['query_labels_one_hot'] = torch.FloatTensor()
        tensors['zero_pad'] = torch.FloatTensor()
        tensors['W_init'] = torch.FloatTensor()
        self.tensors = tensors

    def set_tensors(self, batch):
        support_data, support_labels, query_data, query_labels, Knovel = batch
        self.tensors['support_data'].resize_(support_data.size()).copy_(support_data)
        self.tensors['support_labels'].resize_(support_labels.size()).copy_(support_labels)
        self.tensors['query_data'].resize_(query_data.size()).copy_(query_data)
        self.tensors['query_labels'].resize_(query_labels.size()).copy_(query_labels)

        # one hot labels for support data
        one_hot_size = list(support_labels.size()) + [Knovel.size()[1],]
        labels_unsqueeze = self.tensors['support_labels'].unsqueeze(support_labels.dim())
        self.tensors['support_labels_one_hot'].resize_(one_hot_size). \
            fill_(0).scatter_(support_labels.dim(), labels_unsqueeze, 1)

        # one hot labels for query data
        one_hot_size = list(query_labels.size()) + [Knovel.size()[1],]
        labels_unsqueeze = self.tensors['query_labels'].unsqueeze(query_labels.dim())
        self.tensors['query_labels_one_hot'].resize_(one_hot_size). \
            fill_(0).scatter_(query_labels.dim(), labels_unsqueeze, 1)

        self.tensors['zero_pad'].resize_(one_hot_size).fill_(0)

    def process_batch(self, batch, is_train):
        self.set_tensors(batch)
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
        cur_state['accuracy'] = out['accuracy']
        return cur_state

class GraphConvNet(nn.Module):
    def __init__(self, opt, *args):
        super(GraphConvNet, self).__init__(*args)
        self.emb_size = opt['num_features']
        self.opt = opt
        num_inputs = self.emb_size + opt['num_ways']
        if self.opt['dataset'] == 'miniImageNet':
            self.graph_conv_net = gnn_iclr.GNN_nl(opt, num_inputs, nf=96, J=1)
        elif self.opt['dataset'] == 'omniglot':
            self.graph_conv_net = gnn_iclr.GNN_nl_omniglot(opt, num_inputs, nf=96, J=1)
        else:
            raise NotImplementedError

        if self.opt['dataset'] == 'omniglot':
            self.feature_net = EmbeddingOmniglot(opt)
        elif self.opt['dataset'] == 'miniImageNet':
            self.feature_net = EmbeddingImagenet(opt)
        else:
            raise NotImplementedError

        self.init_optimizer()

    def forward(self, tensors):
        support_data = tensors['support_data']
        query_data = tensors['query_data']
        support_labels_one_hot = tensors['support_labels_one_hot']
        query_labels_one_hot = tensors['query_labels_one_hot']
        query_labels = tensors['query_labels']

        batch_size, num_support_data, num_channels, height, weight = support_data.size()
        batch_size, num_query_data, _, _, _ = query_data.size()

        # process node features
        support_features = self.feature_net(support_data.view(batch_size*num_support_data,
                                                              num_channels, height, weight))
        query_features = self.feature_net(query_data.view(batch_size*num_query_data,
                                                          num_channels, height, weight))
        support_features = support_features.view(batch_size, num_support_data, -1)
        query_features = query_features.view(batch_size, num_query_data, -1)

        combined_features = torch.cat([query_features, support_features], dim=1)
        combined_labels_one_hot = torch.cat([tensors['zero_pad'], support_labels_one_hot], dim=1)
        node_features = torch.cat([combined_features, combined_labels_one_hot], dim=2)

        # feed node features into the graph conv net
        logits = self.graph_conv_net(node_features, tensors)

        # calculate loss
        outputs = F.sigmoid(logits)
        logsoft_prob = F.log_softmax(logits)
        loss = F.nll_loss(logsoft_prob, query_labels.squeeze(1))

        correct_num = torch.sum(torch.argmax(logsoft_prob, 1) == query_labels.squeeze(1)).item()

        out_state = {}
        out_state['loss'] = loss
        out_state['accuracy'] = 1.0 * correct_num / batch_size

        return out_state

    def init_optimizer(self):
        weight_decay = 0
        if self.opt['dataset'] == 'mini_imagenet':
            print('Weight decay '+str(1e-6))
            weight_decay = 1e-6
        self.optimizer = {}
        self.optimizer['feature'] = optim.Adam(self.feature_net.parameters(),
                                                lr=self.opt['lr'], weight_decay=weight_decay)
        self.optimizer['graph_conv'] = optim.Adam(self.graph_conv_net.parameters(),
                                                   lr=self.opt['lr'], weight_decay=weight_decay)

    def zero_grad(self):
        for optimizer in self.optimizer.values():
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizer.values():
            optimizer.step()

    def adjust_lr(self, epoch):
        if (epoch + 1) % 5 == 0:
            for optimizer in self.optimizer.values():
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

    def get_net_state(self):
        res = {}
        res['feature_net_dict'] = self.feature_net.state_dict()
        res['graph_conv_net_dict'] = self.graph_conv_net.state_dict()
        res['optimizer'] = self.optimizer

        return res

    def load_net_state(self, param):
        self.featuere_net.load_state_dict(param['feature_net_dict'])
        self.graph_conv_net.load_state_dict(param['graph_conv_net_dict'])
        self.optimizer = param['optimizer']


class EmbeddingOmniglot(nn.Module):
    ''' In this network the input image is supposed to be 28x28 '''

    def __init__(self, opt):
        super(EmbeddingOmniglot, self).__init__()
        self.emb_size = opt['num_features']
        self.nef = 64

        # input is 1 x 28 x 28
        self.conv1 = nn.Conv2d(3, self.nef, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.nef)
        # state size. (nef) x 14 x 14
        self.conv2 = nn.Conv2d(self.nef, self.nef, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.nef)

        # state size. (1.5*ndf) x 7 x 7
        self.conv3 = nn.Conv2d(self.nef, self.nef, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(self.nef)
        # state size. (2*ndf) x 5 x 5
        self.conv4 = nn.Conv2d(self.nef, self.nef, 3, bias=False)
        self.bn4 = nn.BatchNorm2d(self.nef)
        # state size. (2*ndf) x 3 x 3
        self.fc_last = nn.Linear(3 * 3 * self.nef, self.emb_size, bias=False)
        self.bn_last = nn.BatchNorm1d(self.emb_size)

    def forward(self, inputs):
        e1 = F.max_pool2d(self.bn1(self.conv1(inputs)), 2)
        x = F.leaky_relu(e1, 0.1, inplace=True)

        e2 = F.max_pool2d(self.bn2(self.conv2(x)), 2)
        x = F.leaky_relu(e2, 0.1, inplace=True)

        e3 = self.bn3(self.conv3(x))
        x = F.leaky_relu(e3, 0.1, inplace=True)
        e4 = self.bn4(self.conv4(x))
        x = F.leaky_relu(e4, 0.1, inplace=True)
        x = x.view(-1, 3 * 3 * self.nef)

        output = F.leaky_relu(self.bn_last(self.fc_last(x)))

        return output


class EmbeddingImagenet(nn.Module):
    ''' In this network the input image is supposed to be 28x28 '''

    def __init__(self, opt):
        super(EmbeddingImagenet, self).__init__()
        self.emb_size = opt['num_features']
        self.ndf = 64
        self.opt = opt

        # Input 84x84x3
        self.conv1 = nn.Conv2d(3, self.ndf, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ndf)

        # Input 42x42x64
        self.conv2 = nn.Conv2d(self.ndf, int(self.ndf*1.5), kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(int(self.ndf*1.5))

        # Input 20x20x96
        self.conv3 = nn.Conv2d(int(self.ndf*1.5), self.ndf*2, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.ndf*2)
        self.drop_3 = nn.Dropout2d(0.4)

        # Input 10x10x128
        self.conv4 = nn.Conv2d(self.ndf*2, self.ndf*4, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.ndf*4)
        self.drop_4 = nn.Dropout2d(0.5)

        # Input 5x5x256
        self.fc1 = nn.Linear(self.ndf*4*5*5, self.emb_size, bias=True)
        self.bn_fc = nn.BatchNorm1d(self.emb_size)

    def forward(self, input):
        e1 = F.max_pool2d(self.bn1(self.conv1(input)), 2)
        x = F.leaky_relu(e1, 0.2, inplace=True)
        e2 = F.max_pool2d(self.bn2(self.conv2(x)), 2)
        x = F.leaky_relu(e2, 0.2, inplace=True)
        e3 = F.max_pool2d(self.bn3(self.conv3(x)), 2)
        x = F.leaky_relu(e3, 0.2, inplace=True)
        x = self.drop_3(x)
        e4 = F.max_pool2d(self.bn4(self.conv4(x)), 2)
        x = F.leaky_relu(e4, 0.2, inplace=True)
        x = self.drop_4(x)
        x = x.view(-1, self.ndf*4*5*5)
        output = self.bn_fc(self.fc1(x))

        return output


class MetricNN(nn.Module):
    def __init__(self, opt):
        super(MetricNN, self).__init__()

        self.emb_size = opt['num_features']
        self.opt = opt

        num_inputs = self.emb_size + opt['nKnovel']
        if self.opt['dataset'] == 'mini_imagenet':
            self.gnn_obj = gnn_iclr.GNN_nl(opt, num_inputs, nf=96, J=1)
        elif self.opt['dataset'] == 'omniglot':
            self.gnn_obj = gnn_iclr.GNN_nl_omniglot(opt, num_inputs, nf=96, J=1)
        else:
            raise NotImplementedError

    def forward(self, z, zi_s, labels_yi, zero_pad):
        # Creating WW matrix


        labels_yi = [zero_pad] + labels_yi
        zi_s = [z] + zi_s

        nodes = [torch.cat([zi, label_yi], 1) for zi, label_yi in zip(zi_s, labels_yi)]
        nodes = [node.unsqueeze(1) for node in nodes]
        nodes = torch.cat(nodes, 1)

        logits = self.gnn_obj(nodes).squeeze(-1)
        outputs = F.sigmoid(logits)

        return outputs, logits


class SoftmaxModule():
    def __init__(self):
        self.softmax_metric = 'log_softmax'

    def forward(self, outputs):
        if self.softmax_metric == 'log_softmax':
            return F.log_softmax(outputs)
        else:
            raise(NotImplementedError)


def load_model(model_name, args, io):
    try:
        model = torch.load('checkpoints/%s/models/%s.t7' % (args.exp_name, model_name))
        io.cprint('Loading Parameters from the last trained %s Model' % model_name)
        return model
    except:
        io.cprint('Initiallize new Network Weights for %s' % model_name)
        pass
    return None


def create_models(args):
    print (args.dataset)

    if 'omniglot' == args.dataset:
        enc_nn = EmbeddingOmniglot(args, 64)
    elif 'mini_imagenet' == args.dataset:
        enc_nn = EmbeddingImagenet(args, 128)
    else:
        raise NameError('Dataset ' + args.dataset + ' not knows')
    return enc_nn, MetricNN(args, emb_size=enc_nn.emb_size)
