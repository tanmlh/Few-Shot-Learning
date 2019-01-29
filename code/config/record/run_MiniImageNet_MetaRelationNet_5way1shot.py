# MetaRelationNet 5 way 1 shot MiniImageNet

solver_path = './network/MetaRelationNet.py'

""" Network Options"""
net_opt = {}
net_opt['feature'] = {}
net_opt['feature']['net_name'] = 'FourBlocks'
# net_opt['feature']['num_features'] = 256
net_opt['feature']['userelu'] = True;
net_opt['feature']['in_planes'] = 3
net_opt['feature']['out_planes'] = [64, 64, 128, 128]
net_opt['feature']['num_stages'] = 4

net_opt['relation'] = {}
net_opt['relation']['num_features'] = [6400, 128, 64, 64]

net_opt['meta_relation'] = {}
net_opt['meta_relation']['num_features'] = [128, 64, 64, 1]
net_opt['meta_relation']['ratio'] = [1, 1, 1]

net_opt['use_meta_relation'] = True
net_opt['img_size'] = (3, 84, 84)
net_opt['num_ways'] = 5
net_opt['dataset'] = 'miniImageNet'
net_opt['lr_decay_epoch'] = 20
# net_opt['LUT_lr'] = [(20, 0.01), (40, 0.006), (50, 0.0012), (40, 0.00024)]
net_opt['lr'] = 0.01

conf = {};
conf['solver_name'] = 'MiniImageNet_MetaRelationNet_5way1shot'
conf['net_path'] = solver_path
conf['net_opt'] = net_opt
conf['device_no'] = 1
conf['dataset'] = 'miniImageNet'
# conf['episode_size'] = 2000
# conf['pre_trained'] = {'epoch':98, 'name':'98'}

train_episode_param = {}
train_episode_param['nKnovel'] = 5
train_episode_param['nExemplars'] = 1
train_episode_param['nTestNovel'] = 30
train_batch_size = 8

test_episode_param = {}
test_episode_param['nKnovel'] = 5
test_episode_param['nExemplars'] = 1
test_episode_param['nTestNovel'] = 30
test_batch_size = 8

