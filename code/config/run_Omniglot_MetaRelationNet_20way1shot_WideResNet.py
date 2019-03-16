# MetaRelationNet 20 way 1 shot Omniglot

""" Network Options"""

net_conf = {}

## General network options
net_conf['net_path'] = './network/MetaRelationNet.py'
net_conf['img_size'] = (1, 28, 28)
net_conf['lr'] = 0.1
net_conf['lr_decay_epoch'] = 4
net_conf['decay_ratio'] = 0.8

## Options for feature extraction network 
net_conf['feature'] = {}
net_conf['feature']['in_channels'] = 1
net_conf['feature']['num_classes'] = 4800
net_conf['feature']['net_name'] = 'WideResNet'
net_conf['feature']['drop_rate'] = 0.0
net_conf['feature']['widen_factor'] = 10
net_conf['feature']['depth'] = 28
net_conf['feature']['avg_pool_size'] = 2
net_conf['feature']['block'] = False
# net_conf['feature']['pre_trained'] = '../model/pretrain_MiniImageNet_WideResNet/network_best.pkl'

## Options for relation network
net_conf['relation'] = {}
net_conf['relation']['num_features'] = [640 * 2 * 2, 128, 64]
net_conf['relation']['use_meta_relation'] = True

## Options for meta-relation network
net_conf['meta_relation'] = {}
net_conf['meta_relation']['num_features'] = [128, 64, 1]
net_conf['meta_relation']['ratio'] = [1, 0, 0, 0]


## Options for solver
solver_conf = {}
solver_conf['solver_name'] = 'Omniglot_MetaRelationNet_20way1shot_WideResNet2810_1000'
solver_conf['solver_path'] = './network/MetaRelationNet.py'
solver_conf['best_metric_name'] = 'accuracy_rela'
solver_conf['net_conf'] = net_conf
solver_conf['device_no'] = 3
solver_conf['dataset'] = 'omniglot'
solver_conf['max_epoch'] = 200

## Options for data loader
loader_conf = {}

loader_conf['train_split'] = 'train'
loader_conf['test_split'] = 'test'
loader_conf['epoch_size'] = 2000

train_episode_param = {}
train_episode_param['num_cats'] = 20
train_episode_param['num_sup_per_cat'] = 1
train_episode_param['num_que_per_cat'] = 5

test_episode_param = {}
test_episode_param['num_cats'] = 20
test_episode_param['num_sup_per_cat'] = 1
test_episode_param['num_que_per_cat'] = 5

loader_conf['train_episode_param'] = train_episode_param
loader_conf['test_episode_param'] = test_episode_param
loader_conf['train_batch_size'] = 2
loader_conf['test_batch_size'] = 2

conf = {'net_conf':net_conf, 'solver_conf':solver_conf, 'loader_conf':loader_conf}







