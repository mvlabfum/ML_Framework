import torch
from torch import nn

class Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        prefix = str(self.kwargs.get('prefix', ''))
        if prefix:
            prefix = prefix + '_'
        print('------>', prefix + self.kwargs['netlossfn'], hasattr(self, prefix + self.kwargs['netlossfn']))
        setattr(self, 'forward', getattr(self, prefix + self.kwargs['netlossfn'], getattr(self, f'{prefix}net_loss', None)))
        self.start()
    
    def start(self):
        pass