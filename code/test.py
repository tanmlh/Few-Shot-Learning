import sys
import argparse
import torch
import imp
import pickle
import os

from Omniglot import OmniglotDataset
from MiniImageNet import MiniImageNetDataset
from dataloader import MiniImageNet
from DataLoader import EpisodeLoader, get_loader
import Solver

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', dest='conf_path', type=str, help='configure file path', default=None)
    parser.add_argument('--state_path', dest='state_path', type=str,
                        help='pretrained solver state file', default=None)

    args = parser.parse_args()

    if args.state_path is not None:
        solver = Solver.get_solver_from_pkl(args.state_path)
        solver_conf = solver.conf['solver_conf']
        loader_conf = solver.conf['loader_conf']

    else:
        ## Load parameters ##
        conf = imp.load_source('', args.conf_path).conf
        solver_conf = conf['solver_conf']
        loader_conf = conf['loader_conf']
        solver_path = solver_conf['solver_path']

        ## Create a solver ##
        solver = imp.load_source('', solver_path).get_solver(conf)

    train_episode_param = loader_conf['train_episode_param']
    test_episode_param = loader_conf['test_episode_param']
    train_batch_size = loader_conf['train_batch_size']
    test_batch_size = loader_conf['test_batch_size']
    epoch_size = loader_conf['epoch_size']

    if solver_conf['dataset'] == 'omniglot':
        # train_dataset = OmniglotDataset(is_train=True)
        test_dataset = OmniglotDataset(is_train=False)
    elif solver_conf['dataset'] == 'miniImageNet':
        # train_dataset = MiniImageNetDataset(phase=loader_conf['train_split'])
        test_dataset = MiniImageNetDataset(phase='test')
    else:
        raise NotImplementedError
    ## Create the data loader ##
    test_loader = EpisodeLoader(test_dataset, test_episode_param, test_batch_size, num_workers=6, epoch_size=2000)

    ## Test networks ##
    test_state = solver.test(test_loader)
    print('95% % Confidence Interval for relation: %f% % +- %f% %'
          % (test_state.average()['accuracy'] * 100,
             test_state.confidence_interval()['accuracy'] * 100))

    print('95% % Confidence Interval for meta-relation: %f% % +- %f% %'
          % (test_state.average()['accuracy_meta'] * 100,
             test_state.confidence_interval()['accuracy_meta'] * 100))
