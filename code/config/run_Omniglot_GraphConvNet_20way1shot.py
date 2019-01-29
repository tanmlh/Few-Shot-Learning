solver_path = './network/GraphConvNet.py'

net_opt = {}

"""
feature_model_opt = {}
feature_model_opt['userelu'] = True; feature_model_opt['in_planes'] = 3
feature_model_opt['out_planes'] = [64, 64, 128, 128]; feature_model_opt['num_stages'] = 4
"""

net_opt['img_size'] = (3, 28, 28)
net_opt['num_features'] = 64
net_opt['num_ways'] = 20
net_opt['dataset'] = 'omniglot'
# net_opt['LUT_lr'] = [(10, 0.1), (20, 0.01), (30, 0.001), (40, 0.0001)]
net_opt['lr_decay_epoch'] = 10
net_opt['lr'] = 0.001

conf = {};
conf['solver_name'] = 'Omniglot_GraphConvNet_20way1shot'
conf['net_path'] = solver_path
conf['net_opt'] = net_opt
# conf['pre_trained_epoch'] = 10

train_episode_param = {}
train_episode_param['nKnovel'] = 20
train_episode_param['nExemplars'] = 1
train_episode_param['nTestNovel'] = 1
train_batch_size = 100

test_episode_param = {}
test_episode_param['nKnovel'] = 20
test_episode_param['nExemplars'] = 1
test_episode_param['nTestNovel'] = 1
test_batch_size = 10

