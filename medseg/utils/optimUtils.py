import torch.optim as optim
from typing import Union, List

__all__ = ['get_current_lr']

def get_current_lr(optimizer: optim.Optimizer) -> Union[float, List[float]]:
    """从优化器中获取当前学习率（适配单/多参数组）"""
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    return lrs[0] if len(lrs) == 1 else lrs