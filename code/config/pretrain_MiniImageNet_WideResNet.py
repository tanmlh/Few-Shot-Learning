# pretrain WideResNet on MiniImageNet


""" Network Options"""
net_conf = {}
net_conf['num_classes'] = 64
net_conf['feature'] = {}

net_conf['net_path'] = './network/WideResNet.py'
net_conf['img_size'] = (3, 84, 84)
net_conf['lr'] = 0.01
net_conf['lr_decay_epoch'] = 5
net_conf['decay_ratio'] = 0.8

net_conf['feature'] = {}
net_conf['feature']['num_classes'] = 64
net_conf['feature']['net_name'] = 'WideResNet'
net_conf['feature']['drop_rate'] = 0.0
net_conf['feature']['widen_factor'] = 4
net_conf['feature']['depth'] = 34
net_conf['feature']['avg_pool_size'] = 2
net_conf['feature']['block'] = False
# net_conf['LUT_lr'] = [(20, 0.01), (40, 0.006), (50, 0.0012), (40, 0.00024)]

solver_conf = {};
# conf['pre_trained'] = '../model/MiniImageNet_MetaRelationNet_5way1shot/network_best.pkl'

solver_conf['solver_name'] = 'pretrain_MiniImageNet_WideResNet3404'
solver_conf['solver_path'] = './network/WideResNet.py'
solver_conf['net_conf'] = net_conf
solver_conf['device_no'] = 3
solver_conf['dataset'] = 'miniImageNet'
solver_conf['max_epoch'] = 200
# conf['episode_size'] = 2000

loader_conf = {}
loader_conf['batch_size'] = 64

conf = {'net_conf':net_conf, 'solver_conf':solver_conf, 'loader_conf':loader_conf}
