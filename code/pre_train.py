import sys
import argparse
import torch
import imp
import pickle

from Omniglot import OmniglotDataset
from MiniImageNet import MiniImageNetDataset
from dataloader import MiniImageNet
from DataLoader import BatchLoader

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
    batch_size = conf_module.batch_size

    ## Create a solver ##
    # solver = imp.load_source('', solver_path).get_solver(solver_conf)
    solver = imp.load_source('', solver_path).get_solver(solver_conf)

    ## Prepare the datasets ##

    train_dataset = MiniImageNetDataset(phase='pretrain_train')
    test_dataset = MiniImageNetDataset(phase='pretrain_val')

    ## Create the data loader ##
    train_loader = BatchLoader(train_dataset, batch_size, num_workers=6)
    test_loader = BatchLoader(test_dataset, batch_size, num_workers=6)

    # train_loader = get_loader(train_dataset)
    # test_loader = get_loader(test_dataset)

    ## Train networks ##
    solver.solve(train_loader, test_loader)


