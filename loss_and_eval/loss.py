import torch
from config import *

def loss_func(pred, target, alpha=0.1):
    r"""
    Loss in paper. L_traj + `alpha` * L_node
    L_traj:
        Paper's description:
            - L_traj is the negative Gaussian log-likelihood for the groundtruth future trajectories
        The commonly NLLLoss is for classification problem, but this problem doesn't belong to it.
        Using the literal comprehension, -log(Gaussion(a-b)) is equals MSE, so we use MSE loss.
    L_node:
        Relative to node completion, now we set it to zero.
    Args:
        a: [batch_size, len, dim]
        b:
        alpha: blend factor
    Returns:
        A value.
    """
    # target = torch.stack(target,0).to(config['device'])
    # target = torch.stack(target,0).to(config['device'])
    L_traj = torch.nn.MSELoss().to(config['device'])
    L_node = 0
    return L_traj(pred, target.to(config['device'])) + alpha * L_node
