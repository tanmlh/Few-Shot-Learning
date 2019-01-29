import sys
import os
import logging
import pickle

import torch
from tensorboardX import SummaryWriter
import imp
from tqdm import tqdm

import utils
sys.path.append('network/')
import FeatureModel
from Omniglot import OmniglotDataset
from DataLoader import EpisodeLoader

def get_solver_from_pkl(pkl_path):
    solver_state = pickle.load(open(pkl_path, 'rb'))
    solver = Solver(solver_state['conf'])
    solver.load_net_state(solver_state)


class Solver(object):
    """
    A solver can do network training, evaluation and test
    """
    def __init__(self, conf, *args):
        super(Solver, self).__init__(*args)
        self.conf = conf
        self.model_path = '../model' if 'model_path' not in conf else conf['model_path']
        self.solver_name = conf['solver_name']
        self.max_epoch = 150 if 'max_epoch' not in conf else conf['max_epoch']
        self.cur_epoch = 0
        self.init_networks()
        self.init_best_checkpoint_settings()
        self.init_tensors()
        self.load_to_gpu()
        self.summary_writer = SummaryWriter(log_dir='../runs')

    def init_networks(self):
        """
        Initialize the network model
        """
        net_path_name = self.conf['net_path']
        self.net = imp.load_source('', self.conf['net_path']).create_model(self.conf['net_opt'])

    def solve(self, train_loader, test_loader=None):
        """
        Train and test the network model with the given train and test data loader
        """
        print('Training the network with parameters: ')
        print(self.conf)
        start_epoch = self.cur_epoch
        train_collector = utils.DataCollector()
        test_collector = utils.DataCollector()
        for self.cur_epoch in range(start_epoch, self.max_epoch):
            self.net.adjust_lr(self.cur_epoch)
            train_state = self.train_epoch(train_loader)
            self.update_checkpoint(self.cur_epoch)
            train_collector.update(train_state)
            self.print_state(train_state, self.cur_epoch, True)

            if test_loader is not None:
                eval_state = self.eval_epoch(test_loader).average()
                self.update_best_checkpoint(eval_state, self.cur_epoch)
                test_collector.update(eval_state)
                self.print_state(eval_state, self.cur_epoch, False)

        return train_collector, test_collector

    def test(self, test_loader):
        eval_state = self.eval_epoch(test_loader)
        self.print_state(eval_state.average(), 1, False)
        return eval_state

    def get_solver_state(self):
        state = self.net.get_net_state()
        state['epoch'] = self.cur_epoch
        state['conf'] = self.conf
        return state

    def load_net_state(self, state):
        self.cur_epoch = int(state['epoch'])
        self.net.load_net_state(state)

    def update_checkpoint(self, cur_epoch):
        if not os.path.exists(os.path.join(self.model_path, self.solver_name)):
            os.makedirs(os.path.join(self.model_path, self.solver_name))

        path = os.path.join(self.model_path, self.solver_name,  'network_' + str(cur_epoch) + '.pkl')
        net_state = self.get_solver_state()

        torch.save(net_state, open(path, 'wb'))

        old_path_network = os.path.join(self.model_path, self.solver_name, 'network_' + str(cur_epoch-1) + '.pkl')
        if os.path.isfile(old_path_network) and (cur_epoch) % 10 != 0:
            os.remove(old_path_network)

    def init_best_checkpoint_settings(self):
        self.best_metric_name = 'accuracy'
        self.max_eval_metric = None
        self.best_epoch = None
        self.best_eval_result = None

    def update_best_checkpoint(self, eval_result, cur_epoch):
        assert self.best_metric_name in eval_result.keys()
        cur_eval_metric = eval_result[self.best_metric_name]
        if cur_eval_metric >= self.max_eval_metric or self.max_eval_metric is None:
            self.max_eval_metric = cur_eval_metric
            self.best_epoch = cur_epoch
            self.best_eval_result = eval_result

            path = os.path.join(self.model_path, self.solver_name, 'network_best.pkl')
            net_state = self.get_solver_state()
            torch.save(net_state, open(path, 'wb'))

    def train_epoch(self, train_loader):
        collector = utils.DataCollector()
        for idx, batch in enumerate(tqdm(train_loader(self.cur_epoch))):
            train_state = self.process_batch(batch, True)
            collector.update(train_state)

        return collector.average()

    def eval_epoch(self, eval_loader):
        collector = utils.DataCollector()
        for idx, batch in enumerate(tqdm(eval_loader(self.cur_epoch))):
            eval_state = self.process_batch(batch, False)
            collector.update(eval_state)

        return collector

    def process_batch(self, batch, is_train):
        raise NotImplementedError

    def init_tensors(self):
        self.tensors = {}
        raise NotImplementedError

    def set_tensors(self, batch):
        raise NotImplementedError

    def load_to_gpu(self):
        device_no = self.conf['device_no']
        self.net = self.net.to('cuda:'+str(device_no))
        for key, tensor in self.tensors.items():
            self.tensors[key] = self.tensors[key].to('cuda:'+str(device_no))

    def print_state(self, state, epoch, is_train):
        if is_train:
            print('Training   epoch %d   --> loss: %f | accuracy: %f' % (epoch, state['loss'], state['accuracy']))
            self.summary_writer.add_scalars(os.path.join('log', self.solver_name, 'train_scalars'), state)

        else:
            print('Evaluating epoch %d --> loss: %f | accuracy: %f' % (epoch, state['loss'], state['accuracy']))
            self.summary_writer.add_scalars(os.path.join('log', self.solver_name, 'test_scalars'), state)

    def generate_case(batch):
        pass

    def case_study(self, test_loader):
        case_dir = '../case_study/case'

        for batch in test_loader(0):
            cases = self.generate_case(batch)
            break
        for i, case in enumerate(cases):
            path = case_dir + str(i) + '.pkl'
            pickle.dump(case, open(path, 'wb'))



if __name__ == '__main__':

    conf = {};
    conf['net_path_name'] = './network/PrototypicalNetwork.py'
    conf['solver_name'] = 'Omniglot_ProtoNet_20way1shot'
    # conf['LUT_lr'] = [(5, 0.001), (10, 0.0005), (20, 0.00025), (30, 0.0001), (40, 0.00005)]
    conf['LUT_lr'] = [(5, 0.1), (10, 0.01), (20, 0.001), (30, 0.0001)]
    solver = Solver(conf)

    train_dataset = OmniglotDataset(is_train=True)
    test_dataset = OmniglotDataset(is_train=False)
    train_episode_param = {}
    train_episode_param['nKnovel'] = 20
    train_episode_param['nExemplars'] = 1
    train_episode_param['nTestNovel'] = 100
    train_batch_size = 4

    test_episode_param = {}
    test_episode_param['nKnovel'] = 20
    test_episode_param['nExemplars'] = 1
    test_episode_param['nTestNovel'] = 50
    test_batch_size = 10

    train_loader = EpisodeLoader(train_dataset, train_episode_param,
                                 train_batch_size, num_workers=12)()
    test_loader = EpisodeLoader(test_dataset, test_episode_param,
                                test_batch_size, num_workers=12)()

    solver.solve(train_loader, test_loader)
    """
    dummy_input = torch.rand(13, 3, 28, 28)
    with SummaryWriter(log_dir='./log', comment='Net1') as w:
        w.add_graph(net, (dummy_input))
    """
