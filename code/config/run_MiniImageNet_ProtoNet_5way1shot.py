solver_path = './network/ProtoNet.py'

net_opt = {}
feature_model_opt = {}
feature_model_opt['userelu'] = True; feature_model_opt['in_planes'] = 3
feature_model_opt['out_planes'] = [64, 64, 128, 128]; feature_model_opt['num_stages'] = 4

net_opt['img_size'] = (3, 84, 84)
net_opt['feature_net_opt'] = feature_model_opt
net_opt['proto_head_opt'] = None
net_opt['lr_decay_epoch'] = 20
net_opt['lr'] = 0.01
# net_opt['LUT_lr'] = [(20, 0.01), (40, 0.006), (50, 0.0012), (40, 0.00024)]

conf = {};
conf['solver_name'] = 'MiniImageNet_ProtoNet_5way1shot'
conf['net_path'] = './network/ProtoNet.py'
conf['net_opt'] = net_opt
conf['device_no'] = 0
conf['dataset'] = 'miniImageNet'
# conf['pre_trained'] = 10

train_episode_param = {}
train_episode_param['nKnovel'] = 5
train_episode_param['nExemplars'] = 1
train_episode_param['nTestNovel'] = 30
train_batch_size = 8

test_episode_param = {}
test_episode_param['nKnovel'] = 5
test_episode_param['nExemplars'] = 1
test_episode_param['nTestNovel'] = 15 * 5
test_batch_size = 1

