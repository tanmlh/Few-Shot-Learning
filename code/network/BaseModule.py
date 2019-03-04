import torch

class BaseModule(torch.nn.Module):

    def init_optimizer(self):
        optimizer = {}
        for key, value in self.net.items():
            parameters = filter(lambda x: x.requires_grad, value.parameters())
            if len(parameters) == 0:
                continue
            lr = self.conf['lr'] if 'lr' in self.conf else 0.1
            optimizer[key] = torch.optim.SGD(parameters,
                                             lr=lr, momentum=0.9,
                                             weight_decay=5e-4,
                                             nesterov=True)

            """
            optimizer[key] = torch.optim.Adam(parameters,
                                             lr=0.001,
                                             weight_decay=5e-4)
            """

        self.optimizer = optimizer

    def zero_grad(self):
        for key, value in self.optimizer.items():
            value.zero_grad()

    def step(self):
        for key, value in self.optimizer.items():
            if 'block' not in self.conf[key] or self.conf[key]['block'] is False:
                value.step()

    def adjust_lr(self, cur_epoch):
        if 'LUT_lr' in self.conf:
            for name, optimizer in self.optimizer.items():
                for epoch, lr in self.conf['LUT_lr']:
                    if cur_epoch < epoch:
                        print('learning rate for %s net is changed to %f' % (name, lr))
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        break
        else:
            decay_epoch = self.conf['lr_decay_epoch']
            decay_ratio = self.conf['decay_ratio']
            if cur_epoch != 1 and (cur_epoch-1) % decay_epoch == 0:
                for name, optimizer in self.optimizer.items():
                    for param_group in optimizer.param_groups:
                        print('learning rate for %s net is changed to %f' % (name, param_group['lr'] *
                                                                      decay_ratio))
                        param_group['lr'] = param_group['lr'] * decay_ratio

    def to(self, device):
        for key, value in self.net.items():
            self.net[key] = value.to(device)
        return self

    def get_net_state(self):
        net_state = {}
        optimizer_state = {}
        for key, value in self.net.items():
            net_state[key] = value.state_dict()

        for key, value in self.optimizer.items():
            optimizer_state[key] = value.state_dict()

        state = {}
        state['net_state'] = net_state
        state['optimizer_state'] = optimizer_state
        return state

    def load_net_state(self, state):
        for key, value in self.net.items():
            if key in state['net_state']:
                value.load_state_dict(state['net_state'][key])

        for key, value in self.optimizer.items():
            if key in state['optimizer_state']:
                value.load_state_dict(state['optimizer_state'][key])

    def parameters(self):
        return self.parameters()

    def state_dict(self):
        return self.state_dict()

    def load_state_dict(self, state):
        pass
