import sys
import argparse
import torch
import imp
import pickle

from Omniglot import OmniglotDataset
from MiniImageNet import MiniImageNetDataset
from dataloader import MiniImageNet
from DataLoader import EpisodeLoader, get_loader

sys.path.append('./network/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=str, help='configure file path')
    args = parser.parse_args()

    ## Load parameters ##
    conf_module = imp.load_source('', args.conf)

    ## Extract the parameters ##
    solver_path = conf_module.solver_path
    solver_conf = conf_module.conf
    train_episode_param = conf_module.train_episode_param
    test_episode_param = conf_module.test_episode_param
    train_batch_size = conf_module.train_batch_size
    test_batch_size = conf_module.test_batch_size

    ## Create a solver ##
    # solver = imp.load_source('', solver_path).get_solver(solver_conf)
    solver = imp.load_source('', solver_path).get_solver(solver_conf)

    ## Prepare the datasets ##

    if solver_conf['dataset'] == 'omniglot':
        train_dataset = OmniglotDataset(is_train=True)
        test_dataset = OmniglotDataset(is_train=False)
    else:
        train_dataset = MiniImageNetDataset(phase='train')
        test_dataset = MiniImageNetDataset(phase='val')

    ## Create the data loader ##
    train_loader = EpisodeLoader(train_dataset, train_episode_param, train_batch_size, num_workers=6)
    test_loader = EpisodeLoader(test_dataset, test_episode_param, test_batch_size, num_workers=6)

    # train_loader = get_loader(train_dataset)
    # test_loader = get_loader(test_dataset)

    ## Train networks ##
    solver.solve(train_loader, test_loader)


