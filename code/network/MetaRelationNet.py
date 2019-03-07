import sys
import torch
from torch import nn
import torch.nn.functional as F

sys.path.append('./network')
import FeatureModel
import ConvNet
import ResNet
import WideResNet
from BaseModule import BaseModule


sys.path.append('../')
from Solver import Solver

def get_solver(conf):
    solver = MetaRelationSolver(conf)
    return solver

def get_solver_from_solver_state(solver_state):
    conf = solver_state['conf']
    solver = MetaRelationSolver(conf)
    solver.load_net_state(solver_state)

    return solver

def create_model(conf):
    return MetaRelationModule(conf)

class MetaRelationSolver(Solver):

    def init_tensors(self):
        """
        Define all the tensors that will be used in network computation.
        """
        tensors = {}
        tensors['support_data'] = torch.FloatTensor()
        tensors['support_labels'] = torch.LongTensor()
        tensors['support_labels_one_hot'] = torch.FloatTensor()
        tensors['query_data'] = torch.FloatTensor()
        tensors['query_labels'] = torch.LongTensor()
        tensors['query_labels_real'] = torch.LongTensor()
        tensors['all_ones'] = torch.FloatTensor()
        self.tensors = tensors

    def set_tensors(self, batch):
        """
        Set all the tensors that will be used in network computation.
        """
        support_data, support_labels, query_data, query_labels, query_labels_real = batch
        batch_size = support_data.size(0)
        num_support = support_data.size(1)
        num_query = query_data.size(1)
        num_class = query_labels.max().item() + 1

        self.tensors['support_data'].resize_(support_data.size()).copy_(support_data)
        self.tensors['support_labels'].resize_(support_labels.size()).copy_(support_labels)
        self.tensors['query_data'].resize_(query_data.size()).copy_(query_data)
        self.tensors['query_labels'].resize_(query_labels.size()).copy_(query_labels)
        self.tensors['query_labels_real'].resize_(query_labels_real.size()).copy_(query_labels_real)

        self.tensors['all_ones'].resize_(batch_size, num_query * num_class *
                                         num_class).fill_(1).to(support_data.device)

        # one hot labels for support data
        one_hot_size = list(support_labels.size()) + [num_class,]
        labels_unsqueeze = self.tensors['support_labels'].unsqueeze(support_labels.dim())
        self.tensors['support_labels_one_hot'].resize_(one_hot_size). \
            fill_(0).scatter_(support_labels.dim(), labels_unsqueeze, 1)

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
        out = self.net(self.tensors)

        if is_train:
            out['loss'].backward()
            self.net.step()

        cur_state = {}
        cur_state['loss'] = out['loss'].item()
        cur_state['accuracy'] = out['accuracy_meta']
        cur_state['accuracy_rela'] = out['accuracy_rela']
        cur_state['accuracy_classification'] = out['accuracy_classification']
        return cur_state

    def print_state(self, state, epoch, phase):
        print('%s %d   --> acc_meta: %f | acc_rela: %f | acc_class %f' % (phase,
                                                                         epoch,
                                                                         state['accuracy'],
                                                                         state['accuracy_rela'],
                                                                         state['accuracy_classification']))

class MetaRelationModule(BaseModule):
    def __init__(self, conf, *args):
        super(MetaRelationModule, self).__init__(*args)
        self.conf = conf
        self.net = {}
        if self.conf['feature']['net_name'] == 'ConvNet':
            self.net['feature'] = ConvNet.create_model(conf['feature'])
        elif self.conf['feature']['net_name'] == 'ResNet':
            self.net['feature'] = ResNet.create_model(conf['feature'])
        elif self.conf['feature']['net_name'] == 'WideResNet':
            self.net['feature'] = WideResNet.create_model(conf)
        else:
            raise NotImplementedError

        self.net['relation'] = RelationNet(conf['relation'])
        if conf['relation']['use_meta_relation'] is True:
            self.net['meta_relation'] = MetaRelationNet(conf['meta_relation'])
        else:
            self.net['meta_relation'] = RelationHead(conf['meta_relation'])

        self.init_optimizer()

    def forward(self, tensors):

        ## Initialization ##
        support_data = tensors['support_data'] # bs * (num_class * num_shot) * C * H * W
        support_labels = tensors['support_labels']
        query_data = tensors['query_data'] # bs * num_query * C * H * W
        query_labels = tensors['query_labels']
        query_labels_real = tensors['query_labels_real']


        ## Feature calculation ##
        batch_size, num_support, channel, height, weight = support_data.size()
        # bs * (num_class * num_shot) * num_feature
        support_features = self.net['feature'](support_data.view(batch_size*num_support, channel,
                                                                 height, weight))['feature']
        support_features = support_features.view(batch_size, num_support, -1)

        batch_size, num_query, channel, height, weight = query_data.size()
        # bs * num_query * num_feature
        out = self.net['feature'](query_data.view(batch_size*num_query, channel, height, weight),
                                  query_labels_real.view(batch_size*num_query,))
        query_features = out['feature']
        loss_classification = out['loss']
        accuracy_classification = out['accuracy']

        query_features = query_features.view(batch_size, num_query, -1)

        ## Relation calculation ##
        # bs * num_query * (num_class * num_shot) * num_feature_relation
        relation_features = self.net['relation'](support_features, query_features, tensors)

        ## Meta-relation calculation ##
        # bs * num_query * num_class
        out = self.net['meta_relation'](relation_features, tensors)
        out['loss_classification'] = loss_classification
        ratio = self.conf['meta_relation']['ratio']
        out['loss'] = (out['loss_relation'] * ratio[0]
                       + out['loss_symetry'] * ratio[1]
                       + out['loss_meta_relation'] * ratio[2]
                       + out['loss_classification'] * ratio[3]) / sum(ratio)


        out['accuracy_classification'] = accuracy_classification

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

class RelationNet(torch.nn.Module):
    def __init__(self, conf, *args):
        super(RelationNet, self).__init__(*args)
        num_features = conf['num_features']
        self.conv_layers = torch.nn.Sequential()
        for i in range(len(num_features)-1):
            self.conv_layers.add_module('rela_conv_{}'.format(i),
                                        nn.Conv1d(num_features[i], num_features[i+1], 1, 1))
            self.conv_layers.add_module('rela_bn_{}'.format(i),
                                        nn.BatchNorm1d(num_features[i+1]))
            self.conv_layers.add_module('rela_relu_{}'.format(i),
                                        nn.ReLU(True))
        self.conf = conf
        self.initialization()

    def forward(self, support_features, query_features, tensors):
        """ Size inference """
        support_labels_one_hot = tensors['support_labels_one_hot']

        batch_size, num_support, num_features = support_features.size()
        num_query = query_features.size()[1]

        unsqueeze_query = query_features.unsqueeze(2).repeat(1, 1, num_support, 1)
        unsqueeze_support = support_features.unsqueeze(1).repeat(1, num_query, 1, 1)

        """ Relation calculation """
        if 'use_euler' in self.conf and self.conf['use_euler'] is True:
            raw_relations = -torch.norm(unsqueeze_query - unsqueeze_support, p=2, dim=3).pow(2) / query_features.size(-1)
            # relations = torch.bmm(raw_relations, support_labels_one_hot)
            relations = raw_relations.unsqueeze(1)
            return relations

        ## bs * num_query * (num_class * num_shot) * num_feature ##
        # raw_relations = torch.abs(unsqueeze_query - unsqueeze_support)
        raw_relations = torch.cat([unsqueeze_query, unsqueeze_support], 3)

        relations = raw_relations.view(batch_size, num_query * num_support, -1)
        relations = relations.transpose(1, 2)
        relations = self.conv_layers(relations)

        # bs * num_feature_relation * num_query * num_support
        relations = relations.view(batch_size, -1, num_query, num_support)

        return relations

    def initialization(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




class RelationHead(torch.nn.Module):
    def __init__(self, conf, *args):
        super(RelationHead, self).__init__(*args)

    def forward(self, relations, tensors):
        """ Size inference """
        batch_size, num_features, num_query, num_support = relations.size()
        relations = relations.squeeze(1)

        """ Probability calculation"""
        # bs * (num_class * num_shot) * num_class
        support_labels_one_hot = tensors['support_labels_one_hot']
        query_labels = tensors['query_labels'].view(batch_size * num_query)

        class_prob = torch.bmm(relations, support_labels_one_hot).view(batch_size * num_query, -1)

        """ Loss calculation """
        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(class_prob, query_labels)
        accuracy = (class_prob.argmax(dim=1) == query_labels).sum().item() / (1.0 * batch_size * num_query)

        """ Result processing"""
        out = {}
        out['loss'] = loss
        out['accuracy'] = accuracy

        return out

class MetaRelationNet(nn.Module):
    def __init__(self, conf, *args):
        super(MetaRelationNet, self).__init__(*args)
        self.conf = conf
        num_features = conf['num_features']
        self.conv_layers = torch.nn.Sequential()
        for i in range(len(num_features)-1):
            self.conv_layers.add_module('meta_conv_{}'.format(i),
                                        nn.Conv1d(num_features[i], num_features[i+1], 1, 1))
            self.conv_layers.add_module('meta_bn_{}'.format(i),
                                        nn.BatchNorm1d(num_features[i+1]))
            if i != len(num_features)-2:
                self.conv_layers.add_module('meta_relu_{}'.format(i),
                                            nn.ReLU(True))
            else:
                self.conv_layers.add_module('meta_sigmoid_{}'.format(i),
                                            nn.Sigmoid())
        self.conv_layers2 = torch.nn.Sequential()
        self.conv_layers2.add_module('meta_conv', nn.Conv1d(num_features[0]/2, 1, 1, 1))
        self.conv_layers2.add_module('meta_sigmoid', nn.Sigmoid())


    def forward(self, relations, tensors):

        """ Extract the tensors to be used"""
        # bs * (num_class * num_shot) * num_class
        support_labels_one_hot = tensors['support_labels_one_hot']
        query_labels = tensors['query_labels']
        all_ones = tensors['all_ones']

        """ Size inference """
        # Note: num_support equals to number of relations
        batch_size, num_feature, num_query, num_support = relations.size()
        num_class = support_labels_one_hot.size(-1)
        num_shot = num_support / num_class

        """ Loss Definition """
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss(reduction='elementwise_mean')

        """ Change sample-wise relation to class-wise relation"""
        trans_mat = support_labels_one_hot.unsqueeze(1).repeat(1, num_feature, 1, 1)
        relations = torch.bmm(relations.view(batch_size * num_feature, num_query, num_support),
                              trans_mat.view(batch_size * num_feature, num_support, num_class))
        relations = relations.view(batch_size, num_feature, num_query, num_class) / num_shot

        """ Calculate loss w.r.t relation """
        viewed_relations = relations.view(batch_size, num_feature, -1)
        temp = self.conv_layers2(viewed_relations).squeeze(1).view(batch_size, num_query, num_class)
        class_prob_relation = temp.view(batch_size * num_query, -1)

        query_labels = query_labels.view(batch_size * num_query)
        loss1 = ce_loss(class_prob_relation, query_labels)
        accuracy1 = (class_prob_relation.argmax(dim=1) == query_labels).sum().item() / (1.0 * batch_size * num_query)

        """ Calculate meta-relation """
        relations = relations.unsqueeze(4).repeat(1, 1, 1, 1, num_class)

        # raw_meta_relation: bs * num_feature_relation * num_query * num_class * num_class
        raw_meta_relation = torch.cat([relations, relations.transpose(3, 4)], 1)
        # raw_meta_relation = relations - relations.transpose(3, 4)
        raw_meta_relation = raw_meta_relation.view(batch_size, 2 * num_feature, -1)
        meta_relation = self.conv_layers(raw_meta_relation)
        meta_relation = meta_relation.view(batch_size, num_query, num_class, num_class)

        # to ensure the antisymmetry of relation, the sum of meta_relation and meta_relation_transpose
        # should be close to ones
        meta_relation_transpose = meta_relation.transpose(2, 3)
        sum_meta_relation = (meta_relation + meta_relation_transpose).view(batch_size, -1)

        class_prob = meta_relation.sum(dim=3)
        # class_prob = torch.bmm(meta_relation, support_labels_one_hot)
        class_prob = class_prob.view(batch_size * num_query, -1)

        """ Calculate loss w.r.t. meta-relation"""
        loss2 = mse_loss(sum_meta_relation, all_ones)
        loss3 = ce_loss(class_prob, query_labels)

        accuracy = (class_prob.argmax(dim=1) == query_labels).sum().item() / (1.0 * batch_size * num_query)

        """ Calculate loss w.r.t. overall classification"""

        """ Result processing"""
        conf = self.conf
        out = {}
        # out['loss'] = (loss1 * conf['ratio'][0]
        #                + loss2 * conf['ratio'][1]
        #                + loss3 * conf['ratio'][2]) / (sum(conf['ratio']))
        out['loss_relation'] = loss1
        out['loss_symetry'] = loss2
        out['loss_meta_relation'] = loss3
        out['accuracy_rela'] = accuracy1
        out['accuracy_meta'] = accuracy

        return out

