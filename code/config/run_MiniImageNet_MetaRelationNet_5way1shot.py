# MetaRelationNet 5 way 1 shot MiniImageNet

solver_path = './network/MetaRelationNet.py'

""" Network Options"""
net_opt = {}
net_opt['feature'] = {}
net_opt['feature']['num_classes'] = 64
net_opt['feature']['net_name'] = 'WideResNet'
net_opt['feature']['drop_rate'] = 0.3
# net_opt['feature']['pre_trained'] = '../model/pretrain_MiniImageNet_WideResNet/network_best.pkl'
net_opt['feature']['block'] = False

"""
net_opt['feature']['userelu'] = True;
net_opt['feature']['in_planes'] = 3
net_opt['feature']['out_planes'] = [64, 64, 128, 128]
net_opt['feature']['num_stages'] = 4
"""

net_opt['relation'] = {}
net_opt['relation']['num_features'] = [512, 128, 64]
# net_opt['relation']['num_features'] = [256, 128, 64]

net_opt['use_meta_relation'] = True
net_opt['meta_relation'] = {}
net_opt['meta_relation']['num_features'] = [128, 64, 1]
net_opt['meta_relation']['ratio'] = [1, 1, 2, 3]

net_opt['img_size'] = (3, 84, 84)
net_opt['lr'] = 0.1
net_opt['lr_decay_epoch'] = 5
net_opt['decay_ratio'] = 0.8
# net_opt['LUT_lr'] = [(20, 0.01), (40, 0.006), (50, 0.0012), (40, 0.00024)]

conf = {};
# conf['pre_trained'] = '../model/MiniImageNet_MetaRelationNet_5way1shot/network_best.pkl'
conf['solver_name'] = 'MiniImageNet_MetaRelationNet_5way1shot'
conf['net_path'] = solver_path
conf['net_opt'] = net_opt
conf['device_no'] = 0
conf['dataset'] = 'miniImageNet'
conf['max_epoch'] = 200

epoch_size = 2000

train_episode_param = {}
train_episode_param['num_cats'] = 5
train_episode_param['num_sup_per_cat'] = 1
train_episode_param['num_que_per_cat'] = 5
train_batch_size = 1

test_episode_param = {}
test_episode_param['num_cats'] = 5
test_episode_param['num_sup_per_cat'] = 1
test_episode_param['num_que_per_cat'] = 5
test_batch_size = 1

