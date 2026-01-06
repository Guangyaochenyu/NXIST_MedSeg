import torch
import torch.nn as nn

__all__ = ['CrossEntropy']

class CrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, config, device):
        super(CrossEntropy, self).__init__(
            weight = torch.tensor(
                data    = [config.data.background_weight] + [1/config.data.background_weight] * (config.data.params.num_classes - 1),
                device  = device,
                dtype   = torch.float32,
            )
        )