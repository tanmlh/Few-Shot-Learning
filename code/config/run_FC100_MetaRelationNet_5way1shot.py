# MetaRelationNet 5 way 1 shot FC100



""" Network Options"""

net_conf = {}

## General network options
net_conf['net_path'] = './network/MetaRelationNet.py'
net_conf['img_size'] = (3, 32, 32)
net_conf['lr'] = 0.1
net_conf['lr_decay_epoch'] = 3
net_conf['decay_ratio'] = 0.8


## Options for feature extraction network 

net_conf['feature'] = {}
net_conf['feature']['num_classes'] = 60
net_conf['feature']['net_name'] = 'WideResNet'
net_conf['feature']['drop_rate'] = 0.0
net_conf['feature']['widen_factor'] = 8
net_conf['feature']['depth'] = 22
net_conf['feature']['avg_pool_size'] = 2
net_conf['feature']['block'] = False

# net_conf['feature']['pre_trained'] = '../model/pretrain_MiniImageNet_WideResNet/network_best.pkl'

"""
net_conf['feature'] = {}
net_conf['feature']['num_classes'] = 60
net_conf['feature']['net_name'] = 'ConvNet'
net_conf['feature']['userelu'] = True;
net_conf['feature']['in_planes'] = 3
net_conf['feature']['out_planes'] = [64, 64, 128, 128]
net_conf['feature']['num_stages'] = 4
"""

## Options for relation network
net_conf['relation'] = {}
net_conf['relation']['num_features'] = [512 * 2 * 2, 64, 32]
net_conf['relation']['use_meta_relation'] = True

## Options for meta-relation network
net_conf['meta_relation'] = {}
net_conf['meta_relation']['num_features'] = [64, 1]
net_conf['meta_relation']['ratio'] = [1, 1, 1, 2]


## Options for solver
solver_conf = {}
solver_conf['solver_name'] = 'FC100_MetaRelationNet_5way1shot_ConvNet_1112'
# solver_conf['solver_name'] = 'FC100_MetaRelationNet_5way1shot_WideResNet2804_1112'
solver_conf['solver_path'] = './network/MetaRelationNet.py'
solver_conf['net_conf'] = net_conf
solver_conf['device_no'] = 0
solver_conf['dataset'] = 'FC100'
solver_conf['max_epoch'] = 200
# solver_conf['solver_state__path'] = '../model/MiniImageNet_MetaRelationNet_5way1shot_tune1/network_best.pkl'

## Options for data loader
loader_conf = {}

loader_conf['train_split'] = 'train'
loader_conf['test_split'] = 'val'
loader_conf['epoch_size'] = 2000

train_episode_param = {}
train_episode_param['num_cats'] = 5
train_episode_param['num_sup_per_cat'] = 1
train_episode_param['num_que_per_cat'] = 5

test_episode_param = {}
test_episode_param['num_cats'] = 5
test_episode_param['num_sup_per_cat'] = 1
test_episode_param['num_que_per_cat'] = 5

loader_conf['train_episode_param'] = train_episode_param
loader_conf['test_episode_param'] = test_episode_param
loader_conf['train_batch_size'] = 2
loader_conf['test_batch_size'] = 2

conf = {'net_conf':net_conf, 'solver_conf':solver_conf, 'loader_conf':loader_conf}
