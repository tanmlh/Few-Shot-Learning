import sys
import argparse
import torch
import imp
import pickle

from Omniglot import OmniglotDataset
from MiniImageNet import MiniImageNetDataset
from dataloader import MiniImageNet
from DataLoader import BatchLoader
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

    if solver_conf['dataset'] == 'omniglot':
        train_dataset = OmniglotDataset(is_train=True)
        test_dataset = OmniglotDataset(is_train=False)
    elif solver_conf['dataset'] == 'miniImageNet':
        train_dataset = MiniImageNetDataset(phase='pretrain_train')
        test_dataset = MiniImageNetDataset(phase='pretrain_val')
    else:
        raise NotImplementedError
 
    batch_size = loader_conf['batch_size']
    ## Create the data loader ##
    train_loader = BatchLoader(train_dataset, batch_size, num_workers=6)
    test_loader = BatchLoader(test_dataset, batch_size, num_workers=6)

    # train_loader = get_loader(train_dataset)
    # test_loader = get_loader(test_dataset)

    ## Train networks ##
    solver.solve(train_loader, test_loader)


