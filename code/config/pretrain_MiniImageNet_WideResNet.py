# MetaRelationNet 5 way 1 shot MiniImageNet

solver_path = './network/WideResNet.py'

batch_size = 16

""" Network Options"""
net_opt = {}
net_opt['num_classes'] = 64
net_opt['feature'] = {}

net_opt['img_size'] = (3, 84, 84)
net_opt['lr'] = 0.01
net_opt['lr_decay_epoch'] = 5
net_opt['decay_ratio'] = 0.8
# net_opt['LUT_lr'] = [(20, 0.01), (40, 0.006), (50, 0.0012), (40, 0.00024)]

conf = {};
# conf['pre_trained'] = '../model/MiniImageNet_MetaRelationNet_5way1shot/network_best.pkl'
conf['solver_name'] = 'pretrain_MiniImageNet_WideResNet'
conf['net_path'] = solver_path
conf['net_opt'] = net_opt
conf['device_no'] = 0
conf['dataset'] = 'miniImageNet'
conf['max_epoch'] = 200
# conf['episode_size'] = 2000

