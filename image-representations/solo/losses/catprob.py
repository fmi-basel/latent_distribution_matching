# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn.functional as F
from solo.utils.misc import gather, get_rank
import wandb
import numpy as np

def catprob_loss_func_approx(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1, entropy_multiplier: float = 1.0
) -> torch.Tensor:
    """Computes Probabilistic loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: Prob loss.
    """

    gathered_z1 = gather(z1)
    local_probabilities1 = z1
    probabilities1 = gathered_z1

    gathered_z2 = gather(z2)
    local_probabilities2 = z2
    probabilities2 = gathered_z2

    sim = torch.einsum("ij,ij->i", local_probabilities1, local_probabilities2)
    invariance = torch.mean(sim) 

    cross_probabilities = torch.einsum("ij,ik->jk", probabilities1, probabilities2) / probabilities1.shape[0]
    full_entropy = -torch.sum(cross_probabilities * torch.log(cross_probabilities + 1e-6))

    probabilities = (torch.mean(probabilities1, dim=0) + torch.mean(probabilities2, dim=0))/2
    single_entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-6))

    cond_entropy = full_entropy - single_entropy

    n = gathered_z1.shape[1]
    Z = np.exp(1/temperature) + n - 1
    KL = np.log(n) + np.log(Z) - invariance / temperature - full_entropy

    wandb.log({"invariance": invariance, 
               "full_entropy": full_entropy, 
               "cond_entropy": cond_entropy, 
               "single_entropy": single_entropy, 
               "KL_Div": KL})

    loss = - invariance - temperature * single_entropy * entropy_multiplier
    return loss


def catprob_loss_func_exact(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes Probabilistic loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: Prob loss.
    """

    gathered_z1 = gather(z1)
    local_probabilities1 = z1
    probabilities1 = gathered_z1

    gathered_z2 = gather(z2)
    local_probabilities2 = z2
    probabilities2 = gathered_z2

    sim = torch.einsum("ij,ij->i", local_probabilities1, local_probabilities2)
    invariance = torch.mean(sim) 

    cross_probabilities = torch.einsum("ij,ik->jk", probabilities1, probabilities2) / probabilities1.shape[0]
    full_entropy = -torch.sum(cross_probabilities * torch.log(cross_probabilities + 1e-6))

    probabilities = (torch.mean(probabilities1, dim=0) + torch.mean(probabilities2, dim=0))/2
    single_entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-6))

    cond_entropy = full_entropy - single_entropy

    n = gathered_z1.shape[1]
    Z = np.exp(1/temperature) + n - 1
    KL = np.log(n) + np.log(Z) - invariance / temperature - full_entropy

    wandb.log({"invariance": invariance, 
               "full_entropy": full_entropy, 
               "cond_entropy": cond_entropy, 
               "single_entropy": single_entropy, 
               "KL_Div": KL})

    loss = - invariance - temperature * full_entropy 
    return loss

def catprob_loss_func_MI(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes Probabilistic loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: Prob loss.
    """

    gathered_z1 = gather(z1)
    local_probabilities1 = z1
    probabilities1 = gathered_z1

    gathered_z2 = gather(z2)
    local_probabilities2 = z2
    probabilities2 = gathered_z2

    sim = torch.einsum("ij,ij->i", local_probabilities1, local_probabilities2)
    invariance = torch.mean(sim) 

    cross_probabilities = torch.einsum("ij,ik->jk", probabilities1, probabilities2) / probabilities1.shape[0]
    full_entropy = -torch.sum(cross_probabilities * torch.log(cross_probabilities + 1e-6))

    probabilities = (torch.mean(probabilities1, dim=0) + torch.mean(probabilities2, dim=0))/2
    single_entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-6))

    cond_entropy = full_entropy - single_entropy

    MI = single_entropy - cond_entropy

    n = gathered_z1.shape[1]
    Z = np.exp(1/temperature) + n - 1
    KL = np.log(n) + np.log(Z) - invariance / temperature - full_entropy

    wandb.log({"invariance": invariance, 
               "full_entropy": full_entropy, 
               "cond_entropy": cond_entropy, 
               "single_entropy": single_entropy, 
               "MI": MI,
               "KL_Div": KL})

    loss = - MI
    return loss


def catprob_loss_func(
        z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1, end_temperature: float = 0.1,
        type: str = "approx", entropy_multiplier: float = 1.0
) -> torch.Tensor:
    

    if type == "approx":
        return catprob_loss_func_approx(z1, z2, temperature, entropy_multiplier)
    elif type == "exact":
        return catprob_loss_func_exact(z1, z2, temperature)
    elif type == "MI":
        return catprob_loss_func_MI(z1, z2, temperature)
    
    raise ValueError(f"Loss type {type} not supported.")