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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=str, help='configure file path')
    args = parser.parse_args()

    ## Load parameters ##
    conf_module = imp.load_source('', args.conf)

    ## Extract the parameters ##
    solver_path = conf_module.solver_path
    solver_conf = conf_module.conf
    solver_name = solver_conf['solver_name']
    train_episode_param = conf_module.train_episode_param
    test_episode_param = conf_module.test_episode_param
    train_batch_size = conf_module.train_batch_size
    test_batch_size = conf_module.test_batch_size
    model_path = '../model'
    solver_conf['pre_trained'] = os.path.join(model_path, solver_name, 'network_best.pkl')

    ## Create a solver ##
    solver = imp.load_source('', solver_path).get_solver(solver_conf)

    ## Prepare the datasets ##
    if solver_conf['dataset'] == 'omniglot':
        test_dataset = OmniglotDataset(is_train=False)
    else:
        test_dataset = MiniImageNetDataset(phase='test')


    ## Create the data loader ##
    test_loader = EpisodeLoader(test_dataset, test_episode_param, test_batch_size, num_workers=6,
                                epoch_size=10000)

    ## Test networks ##
    test_state = solver.test(test_loader)
    print('95% % Confidence Interval for relation: %f% % +- %f% %'
          % (test_state.average()['accuracy'] * 100,
             test_state.confidence_interval()['accuracy'] * 100))

    print('95% % Confidence Interval for meta-relation: %f% % +- %f% %'
          % (test_state.average()['accuracy_meta'] * 100,
             test_state.confidence_interval()['accuracy_meta'] * 100))
