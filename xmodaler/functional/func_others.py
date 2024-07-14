# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import math
import torch

def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]

def noise_injection(x, variance = 0.001, device = 'cuda:0') -> torch.Tensor:
    """
    Args:
        x: tensor with a shape of (batch_size, clip_hidden_size), prefix
        variance: the variance of noise
    Return:
        prefix with noise
    """
    if variance == 0.0:
        return x
    std = math.sqrt(variance)
    # normalization
    x = torch.nn.functional.normalize(x, dim = -1)
    # adding noise
    x = x + (torch.randn(x.shape, device = device) * std)

    return torch.nn.functional.normalize(x, dim = -1)