

import torch
import math
import torch.nn.functional as F
from solo.utils.misc import gather, get_rank
import wandb
import numpy as np


def gauss_similarity(mean1: torch.Tensor, mean2: torch.Tensor, 
                     prediction_precision: float = 1.0,
) -> torch.Tensor:
    """Computes the log gaussian similarity of two latents averaged over the recognition distributions. 

    Args:
        mean1 (torch.Tensor): mean of the first latent with dims [B, D].
        mean2 (torch.Tensor): mean of the second latent with dims [B, D].
        cov1 (torch.Tensor): diagonal of the covariance matrix of the first Gaussian with dims [B, D].
        cov2 (torch.Tensor): diagonal of the covariance matrix of the second Gaussian with dims [B, D].
        prediction_precision (float): precision of distance between related pairs.
        prior_precision (float): precision of the prior on latents (centered at 0).

    Return:
        torch.Tensor: Gaussian similarity.
    """

    mean_diff = mean1 - mean2
    diff_square = torch.mean(mean_diff * mean_diff, dim=1) # [B]

    similarity = - prediction_precision * diff_square 

    return similarity 

def diagonal_gaussian_pdf(x, mean, kernel_precision):
    """Compute the probability density function of a diagonal Gaussian.
    
    Args:
        x (torch.Tensor): input tensor with dims [D].
        mean (torch.Tensor): mean of the Gaussian with dims [D].
        cov (torch.Tensor): diagonal of the covariance matrix of the Gaussian with dims [D].
    
    Returns:
        torch.Tensor: probability density function of the Gaussian.
    """
    
    # Calculate the exponential term
    diff = x - mean  # [D]
    scaled_diff = diff * diff * kernel_precision  
    exp_term = torch.exp(-torch.mean(scaled_diff))  # Sum across dimension D
    
    # Combine to get PDF
    pdf = exp_term # norm term can be omitted for entropy calculations, since it becomes additive in log space
    
    return pdf






def contrastive_dual_entropy_term(mean1, mean2, kernel_precision):
    """
    Compute the contrastive entropy lower bound. This is the log-sum term in the contrastive
    loss, which is a lower bound on the entropy of the joint distribution of two Gaussians.

    Args:
        mean1 (torch.Tensor): mean of the first Gaussian with dims [B, D].
        mean2 (torch.Tensor): mean of the second Gaussian with dims [B, D].
        kernel_precision (torch.Tensor): kernel width for the Gaussians.

    Returns:
        entropy: Scalar tensor (averaged over batch) representing the entropy term
    """
    mean = torch.cat([mean1, mean2], dim=1)  # Shape: [B, 2D]
    entropy = contrastive_single_entropy_term(mean, kernel_precision)
    
    return entropy

def contrastive_single_entropy_term(mean, kernel_precision):
    """
    Compute the contrastive entropy lower bound. This is the log-sum term in the contrastive
    loss, which is a lower bound on the entropy of a Gaussian Mixture.

    Args:
        mean (torch.Tensor): mean of the Gaussians with dims [B, D].
        kernel_precision (torch.Tensor): kernel width for the Gaussians.

    Returns:
        entropy: Scalar tensor (averaged over batch) representing the entropy term
    """
    D = mean.shape[1]

    # Vectorize over the first batch dimension 
    vectorized_inner = torch.vmap(diagonal_gaussian_pdf, in_dims=(0, None, None))
    # Vectorize over the second batch dimension 
    vectorized_outer = torch.vmap(vectorized_inner, in_dims=(None, 0, None))

    # Compute pairwise samples
    samples = vectorized_outer(mean, mean, kernel_precision)  # Shape: [batch_size, batch_size]
    
    # Compute the log-sum term
    sum_term = torch.sum(samples, dim=1)
    entropy = -torch.mean(torch.log(sum_term + 1e-8)) 
    
    return entropy






def knn(x, k=3, p=2, last_only=False):
    """Find k_neighbors-nearest neighbor distances from y for each example in a minibatch x.
    :param x: tensor of shape [T, N]
    :param k: the (k_neighbors+1):th nearest neighbor
    :param p: p-norm to use
    :param last_only: use only the last knn vs. all of them
    :return: knn distances of shape [T, k_neighbors] or [T, 1] if last_only
    """

    distmat = torch.cdist(x, x, p=p)  # shape [T, T]
    knn, _ = torch.topk(distmat, k + 1, largest=False)
    knn = knn[:, 1:] # discard self-distance

    if last_only:
        knn = knn[:, -1:]  # k_neighbors:th distance only

    return knn

def kozachenko_leonenko_single_entropy(x, k=3, p=2, eps=1e-8, last_only=True):
    """Entropy estimator for batch x~p(x).
        :param x: prediction; shape [T, N]
        :param k:
        :return: scalar
        """
    if type(x) is np.ndarray:
        x = torch.tensor(x.astype(np.float32))

    nns_xx = knn(x, k=k, p=p, last_only=last_only)
    nns_xx = nns_xx - torch.clip(nns_xx, max=0.0) # ensure non-negative
    nns_xx = torch.clip(nns_xx, max=1e6) # avoid inf

    # deal with outliers
    # these representations are already well separated, so clip extreme distances
    upper_bound = torch.quantile(nns_xx, 0.9).detach()
    nns_xx = torch.clip(nns_xx, max=upper_bound)

    ent = torch.log(nns_xx + eps).mean() 

    return ent

def kozachenko_leonenko_dual_entropy(x1, x2, k=3, p=2, eps=1e-8, last_only=True):
    """Entropy estimator for joint batch (x1,x2)~p(x1,x2).
        :param x1: prediction; shape [T, N]
        :param x2: prediction; shape [T, N]
        :param k:
        :return: scalar
        """

    x = torch.cat([x1, x2], dim=1)

    return kozachenko_leonenko_single_entropy(x, k=k, p=p, eps=eps, last_only=last_only)





def gaussprob_loss_func_single(
    z1: torch.Tensor, z2: torch.Tensor, prediction_precision: float = 0.1, kernel_precision: float = 0.1, entropy_multiplier: float = 1.0
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

    means1 = gather(z1)
    means2 = gather(z2)

    sim = gauss_similarity(means1, means2, prediction_precision)
    invariance = torch.mean(sim) 

    full_entropy = contrastive_dual_entropy_term(means1, means2, kernel_precision)
    single_entropy = 0.5 * (contrastive_single_entropy_term(means1, kernel_precision) + contrastive_single_entropy_term(means2, kernel_precision))
    cond_entropy = full_entropy - single_entropy

    wandb.log({"invariance": invariance, "full_entropy": full_entropy, "cond_entropy": cond_entropy, "single_entropy": single_entropy})

    loss = - invariance - single_entropy * entropy_multiplier 
    return loss, single_entropy * entropy_multiplier


def gaussprob_loss_func_dual(
    z1: torch.Tensor, z2: torch.Tensor, prediction_precision: float = 0.1, kernel_precision: float = 0.1
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

    means1 = gather(z1)
    means2 = gather(z2)

    sim = gauss_similarity(means1, means2, prediction_precision)
    invariance = torch.mean(sim) 

    full_entropy = contrastive_dual_entropy_term(means1, means2, kernel_precision)
    single_entropy = 0.5 * (contrastive_single_entropy_term(means1, kernel_precision) + contrastive_single_entropy_term(means2, kernel_precision))
    cond_entropy = full_entropy - single_entropy

    wandb.log({"invariance": invariance, "full_entropy": full_entropy, "cond_entropy": cond_entropy, "single_entropy": single_entropy})

    loss = - invariance - full_entropy 
    return loss, full_entropy




def gaussprob_loss_func_dual_knn(
    z1: torch.Tensor, z2: torch.Tensor,
    prediction_precision: float = 0.1
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

    means1 = gather(z1)
    means2 = gather(z2)

    sim = gauss_similarity(means1, means2, prediction_precision)
    invariance = torch.mean(sim) 

    full_entropy = kozachenko_leonenko_dual_entropy(means1, means2, k=3, p=2, eps=1e-8, last_only=True)
    single_entropy = 0.5 * (kozachenko_leonenko_single_entropy(means1, k=3, p=2, eps=1e-8, last_only=True) + kozachenko_leonenko_single_entropy(means2, k=3, p=2, eps=1e-8, last_only=True))
    cond_entropy = full_entropy - single_entropy

    wandb.log({"invariance": invariance, "full_entropy": full_entropy, "cond_entropy": cond_entropy, "single_entropy": single_entropy})

    loss = - invariance - full_entropy 
    return loss, full_entropy



def gaussprob_loss_func_single_knn(
    z1: torch.Tensor, z2: torch.Tensor, 
    prediction_precision: float = 0.1, entropy_multiplier: float = 1.0
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

    means1 = gather(z1)
    means2 = gather(z2)

    sim = gauss_similarity(means1, means2, prediction_precision)
    invariance = torch.mean(sim) 

    full_entropy = kozachenko_leonenko_dual_entropy(means1, means2, k=3, p=2, eps=1e-8, last_only=True)
    single_entropy = 0.5 * (kozachenko_leonenko_single_entropy(means1, k=3, p=2, eps=1e-8, last_only=True) + kozachenko_leonenko_single_entropy(means2, k=3, p=2, eps=1e-8, last_only=True))
    cond_entropy = full_entropy - single_entropy

    wandb.log({"invariance": invariance, "full_entropy": full_entropy, "cond_entropy": cond_entropy, "single_entropy": single_entropy})

    loss = - invariance - single_entropy * entropy_multiplier
    return loss, single_entropy * entropy_multiplier





    

def gaussprob_loss_func(
        z1: torch.Tensor, z2: torch.Tensor,
        prediction_precision: float = 0.1,
        kernel_precision: float = 0.1,
        type: str = "dual_sample", entropy_multiplier: float = 1.0
) -> torch.Tensor:
    
    if type == "single_sample":
        return gaussprob_loss_func_single(z1, z2, 
                                          prediction_precision, kernel_precision, entropy_multiplier)
    elif type == "dual_sample":
        return gaussprob_loss_func_dual(z1, z2, prediction_precision, kernel_precision)
    elif type == "single_knn":
        return gaussprob_loss_func_single_knn(z1, z2,  
                                             prediction_precision, entropy_multiplier)
    elif type == "dual_knn":
        return gaussprob_loss_func_dual_knn(z1, z2, prediction_precision)

    raise ValueError(f"Loss type {type} not supported.")