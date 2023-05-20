import torch
from torch import nn
from utils.pt.loss import Loss

class NET_Loss(Loss):
    def __init__(self, **kwargs):
        kwargs['prefix'] = kwargs.get('prefix', 'mse')
        super().__init__(**kwargs)

    def start(self):
        self.mse = nn.MSELoss()

    def mse_net_loss(self, y, t):
        print(y)
        print(t)
        assert False
        loss = self.mse(y, t)
        log = {
            'loss': loss.clone().detach().mean(),
        }
        return loss, log