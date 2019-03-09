import sys
import argparse
import torch
import imp
import pickle

from Omniglot import OmniglotDataset
from MiniImageNet import MiniImageNetDataset
from FC100 import FC100Dataset
from dataloader import MiniImageNet
from DataLoader import EpisodeLoader, get_loader
import Solver

sys.path.append('./network/')

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
        train_dataset = OmniglotDataset(is_train=True)
        val_dataset = OmniglotDataset(is_train=False)
    elif solver_conf['dataset'] == 'miniImageNet':
        train_dataset = MiniImageNetDataset(phase='train')
        val_dataset = MiniImageNetDataset(phase='val')
        test_dataset = MiniImageNetDataset(phase='test')
    elif solver_conf['dataset'] == 'FC100':
        train_dataset = FC100Dataset(phase='train')
        val_dataset = FC100Dataset(phase='val')
        test_dataset = FC100Dataset(phase='test')
    else:
        raise NotImplementedError

    ## Create the data loader ##
    train_loader = EpisodeLoader(train_dataset, train_episode_param, train_batch_size,
                                 num_workers=6, epoch_size=epoch_size)
    val_loader = EpisodeLoader(val_dataset, test_episode_param, test_batch_size, num_workers=6,
                               epoch_size=epoch_size)
    test_loader = EpisodeLoader(test_dataset, test_episode_param, test_batch_size, num_workers=6,
                                epoch_size=epoch_size)

    # temp = next(iter(train_loader(0)))
    # train_loader = get_loader(train_dataset)
    # test_loader = get_loader(test_dataset)

    ## Train networks ##
    # solver.solve(train_loader, val_loader)
    solver.solve(train_loader, val_loader, test_loader)
