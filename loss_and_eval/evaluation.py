import torch
from config import *

r"""
Reference:
https://eval.ai/web/challenges/challenge-page/454/evaluation
https://github.com/argoai/argoverse-api/blob/master/argoverse/evaluation/eval_forecasting.py
"""

def get_ADE(pred, target):
    r"""
    Calculate Average Displacement Error(ADE).
    Args:
        a: [batch_size, len, dim]
        b:
    Returns: 
        ADE, \frac{1}{n} \sum sqrt((A_x_i-B_x_i)^2 + (A_y_i-B_y_i)^2)
        [batch_size, 1]
    """
    # target = torch.stack(target, 0).to(config['device'])
    target = target.to(config['device'])
    assert pred.shape == target.shape
    tmp = torch.sqrt(torch.sum((pred - target) ** 2, dim=2)) # [batch_size, len]
    ade = torch.mean(tmp, dim=1, keepdim=True) # [batch_size, 1]
    return ade 

def get_FDE(pred, target):
    r"""
    Calculate Final Displacement Error(FDE).
    Args:
        a: [batch_size, len, dim]
        b:
    Returns: 
        FDE, [batch_size, 1]
    """
    target = torch.stack(target, 0).to(config['device'])
    assert pred.shape == target.shape  #batch_size, 30, 2
    pred = pred[:, -1, :] # [batch_size, dim]
    target = target[:, -1, :]
    fde = torch.sqrt(torch.sum((pred - target) ** 2, dim=1, keepdim=True)) # [batch_size, 1]
    return fde 

def get_DE(pred, target, t_list):
    r"""
    Calculate Displacement Error(DE) at time `t` in `t_list`.
    Args:
        a: [batch_size, len, dim]
        b:
        t_list (list): len(t_list)=n
    Returns: 
        DE, [batch_size, n]
    """
    target = torch.stack(target, 0).to(config['device'])
    t_tensor = torch.tensor(t_list)
    pred = torch.index_select(pred, 1, t_tensor) # [batch_size, n, dim]
    target = torch.index_select(target, 1, t_tensor)
    de = torch.sqrt(torch.sum((pred - target) ** 2, dim=2))
    return de