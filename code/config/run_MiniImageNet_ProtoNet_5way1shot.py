
""" Network Options"""

net_conf = {}

## General network options
net_conf['net_path'] = './network/ProtoNet.py'
net_conf['img_size'] = (3, 84, 84)
net_conf['lr'] = 0.1
net_conf['lr_decay_epoch'] = 4
net_conf['decay_ratio'] = 0.8


## Options for feature extraction network 

net_conf['feature'] = {}
net_conf['feature']['num_classes'] = 64
net_conf['feature']['net_name'] = 'WideResNet'
net_conf['feature']['drop_rate'] = 0.0
net_conf['feature']['widen_factor'] = 4
net_conf['feature']['depth'] = 40
net_conf['feature']['avg_pool_size1'] = 5
net_conf['feature']['avg_pool_size2'] = 2
net_conf['feature']['block'] = False

net_conf['head'] = {}
net_conf['ratio'] = [1, 2]

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

## Options for solver
solver_conf = {}
solver_conf['solver_name'] = 'MiniImageNet_ProtoNet_5way1shot_WideResNet4004'
solver_conf['solver_path'] = './network/ProtoNet.py'
solver_conf['net_conf'] = net_conf
solver_conf['device_no'] = 0
solver_conf['dataset'] = 'miniImageNet'
solver_conf['max_epoch'] = 200
solver_conf['index_names'] = ['accuracy']
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
