

import torch
from solo.utils.misc import gather
import wandb
import numpy as np


def sphere_similarity(mean1: torch.Tensor, mean2: torch.Tensor, 
                     gamma1: torch.Tensor, gamma2: torch.Tensor,
                     prediction_precision: float = 1.0
) -> torch.Tensor:
    """Computes the spherical similarity of two latents averaged over the recognition vMF distributions. 

    Args:
        mean1 (torch.Tensor): mean of the first latent with dims [B, D].
        mean2 (torch.Tensor): mean of the second latent with dims [B, D].
        gamma1 (torch.Tensor): precision of the first latent with dims [B].
        gamma2 (torch.Tensor): precision of the second latent with dims [B].
        prediction_precision (float): precision of distance between related pairs.

    Return:
        torch.Tensor: similarity of the two latents.
    """

    sim = torch.einsum("ij,ij->i", mean1, mean2)
    sim = sim * prediction_precision

    return sim

def mises_fisher_pdf_approx(x, mean, gamma):
    """
    Compute the probability density function of the von Mises-Fisher distribution.
    This is the unnormalized pdf, which is computationally less expensive.
    All constant terms are omitted, assuming gamma is unchanging.
    
    Args:
        x (torch.Tensor): point at which to evaluate the pdf with dims [D].
        mean (torch.Tensor): mean of the distribution with dims [D].
        gamma (torch.Tensor): precision of the distribution with dims [].
    
    Returns:
        pdf: Scalar tensor representing the pdf of the distribution at the point x.
    """

    inner_product = torch.dot(x, mean)
    log_sim = gamma * inner_product
    
    pdf = torch.exp(log_sim)
    
    return pdf

def contrastive_dual_entropy_term(mean1, mean2, gamma1, gamma2):
    """
    Compute the contrastive entropy. This is the log-sum term in the contrastive
    loss, which is an approximation of the entropy of the joint distribution of two vMF ignoring constants.

    Args:
        mean1 (torch.Tensor): mean of the first vMF with dims [B, D].
        mean2 (torch.Tensor): mean of the second vMF with dims [B, D].
        gamma1 (torch.Tensor): precision of the first vMF with dims [B].
        gamma2 (torch.Tensor): precision of the second vMF with dims [B].

    Returns:
        entropy: Scalar tensor (averaged over batch) representing the entropy term
    """
    
    means = torch.cat([mean1, mean2], dim=1)  # Shape: [B, 2*D]
    gammas = gamma1  # Shape: [B]
    entropy = contrastive_single_entropy_term(means, gammas)

    return entropy

def contrastive_single_entropy_term(mean, gamma):
    """
    Compute the contrastive entropy. This is the log-sum term in the contrastive
    loss, which is an approximation of the entropy of a vMF mixture ignoring constants.

    Args:
        mean (torch.Tensor): mean of the vMF with dims [B, D].
        gamma (torch.Tensor): precision of the vMF with dims [B].

    Returns:
        entropy: Scalar tensor (averaged over batch) representing the entropy term
    """

    pdf_func = mises_fisher_pdf_approx
    # Vectorize over the first batch dimension 
    vectorized_inner = torch.vmap(pdf_func, in_dims=(0, None, None))
    # Vectorize over the second batch dimension 
    vectorized_outer = torch.vmap(vectorized_inner, in_dims=(None, 0, 0))

    # Compute pairwise samples
    samples = vectorized_outer(mean, mean, gamma)  # Shape: [batch_size, batch_size]
    
    # Compute the log-sum term
    sum_term = torch.sum(samples , dim=1) 
    entropy = -torch.mean(torch.log(sum_term)) 
    
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




def log_det_expansion(cov):
    """Computes the log-determinant of a covariance matrix using its Taylor expansion.

    Args:
        cov (torch.Tensor): DxD covariance matrix.
        eps (float): small value to avoid numerical issues.
    Returns:
        torch.Tensor: log-determinant of the covariance matrix.
    """
    vars = torch.diag(cov)
    D = cov.size(0)
    corrs = cov**2 / (torch.outer(vars, vars) + 1e-9)
    diag = torch.eye(D, device=cov.device)
    diag_term = torch.log(vars + 1e-9).sum()
    off_diag_term = -0.5 * corrs[~diag.bool()].pow_(2).sum()
    return diag_term + off_diag_term
    



def sphereprob_loss_func_dual(
    z1: torch.Tensor, z2: torch.Tensor, gamma1: torch.Tensor, gamma2: torch.Tensor,
    prediction_precision: float = 0.1, 
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
    assert not torch.isnan(z1).any() and not torch.isinf(z1).any(), "NaNs/Infs in z1"
    assert not torch.isnan(z2).any() and not torch.isinf(z2).any(), "NaNs/Infs in z2"

    means1 = gather(z1)
    means2 = gather(z2)
    gamma1 = gather(gamma1)
    gamma2 = gather(gamma2)

    sim = sphere_similarity(means1, means2, gamma1, gamma2, prediction_precision)
    invariance = torch.mean(sim) 

    full_entropy = contrastive_dual_entropy_term(means1, means2, gamma1, gamma2)
    single_entropy = 0.5 * (contrastive_single_entropy_term(means1, gamma1) + contrastive_single_entropy_term(means2, gamma2))
    cond_entropy = full_entropy - single_entropy

    wandb.log({"invariance": invariance, "full_entropy": full_entropy, "cond_entropy": cond_entropy, "single_entropy": single_entropy})


    assert not torch.isnan(full_entropy).any() and not torch.isinf(full_entropy).any(), "NaNs/Infs in z1"
    assert not torch.isnan(invariance).any() and not torch.isinf(invariance).any(), "NaNs/Infs in z2"

    loss = - invariance - full_entropy 
    return loss, full_entropy

def sphereprob_loss_func_dual_gauss(
    z1: torch.Tensor, z2: torch.Tensor, gamma1: torch.Tensor, gamma2: torch.Tensor,
    prediction_precision: float = 0.1, 
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
    assert not torch.isnan(z1).any() and not torch.isinf(z1).any(), "NaNs/Infs in z1"
    assert not torch.isnan(z2).any() and not torch.isinf(z2).any(), "NaNs/Infs in z2"

    eps = torch.diag(1e-6 * torch.eye(z1.size(1), device=z1.device))

    means1 = gather(z1)
    means2 = gather(z2)
    gamma1 = gather(gamma1)
    gamma2 = gather(gamma2)

    sim = sphere_similarity(means1, means2, gamma1, gamma2, prediction_precision)
    invariance = torch.mean(sim) 

    means1 = means1 - means1.mean(dim=0)
    means2 = means2 - means2.mean(dim=0)
    means = torch.cat([means1, means2], dim=0)
    N, D = means.size()
    cov = (means.T @ means) / (N - 1)
    cov1 = (means1.T @ means1) / (N//2 - 1) 
    cov2 = (means2.T @ means2) / (N//2 - 1)

    full_entropy = log_det_expansion(cov + eps) / D
    single_entropy = 0.5 * (log_det_expansion(cov1 + eps) + log_det_expansion(cov2 + eps)) / D
    cond_entropy = full_entropy - single_entropy

    wandb.log({"invariance": invariance, "full_entropy": full_entropy, "cond_entropy": cond_entropy, "single_entropy": single_entropy})


    assert not torch.isnan(full_entropy).any() and not torch.isinf(full_entropy).any(), "NaNs/Infs in z1"
    assert not torch.isnan(invariance).any() and not torch.isinf(invariance).any(), "NaNs/Infs in z2"

    loss = - invariance - full_entropy 
    return loss, full_entropy

def sphereprob_loss_func_dual_knn(
    z1: torch.Tensor, z2: torch.Tensor, gamma1: torch.Tensor, gamma2: torch.Tensor,
    prediction_precision: float = 0.1, 
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
    assert not torch.isnan(z1).any() and not torch.isinf(z1).any(), "NaNs/Infs in z1"
    assert not torch.isnan(z2).any() and not torch.isinf(z2).any(), "NaNs/Infs in z2"

    means1 = gather(z1)
    means2 = gather(z2)
    gamma1 = gather(gamma1)
    gamma2 = gather(gamma2)

    sim = sphere_similarity(means1, means2, gamma1, gamma2, prediction_precision)
    invariance = torch.mean(sim) 

    full_entropy = kozachenko_leonenko_dual_entropy(means1, means2)
    single_entropy = 0.5 * (kozachenko_leonenko_single_entropy(means1) + kozachenko_leonenko_single_entropy(means2))
    cond_entropy = full_entropy - single_entropy

    wandb.log({"invariance": invariance, "full_entropy": full_entropy, "cond_entropy": cond_entropy, "single_entropy": single_entropy})


    assert not torch.isnan(full_entropy).any() and not torch.isinf(full_entropy).any(), "NaNs/Infs in z1"
    assert not torch.isnan(invariance).any() and not torch.isinf(invariance).any(), "NaNs/Infs in z2"

    loss = - invariance - full_entropy 
    return loss, full_entropy

def sphereprob_loss_func_single(
    z1: torch.Tensor, z2: torch.Tensor, gamma1: torch.Tensor, gamma2: torch.Tensor,
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
    gamma1 = gather(gamma1)
    gamma2 = gather(gamma2)

    sim = sphere_similarity(means1, means2, gamma1, gamma2, prediction_precision)
    invariance = torch.mean(sim) 

    full_entropy = contrastive_dual_entropy_term(means1, means2, gamma1, gamma2) 
    single_entropy = 0.5 * (contrastive_single_entropy_term(means1, gamma1) + contrastive_single_entropy_term(means2, gamma2))
    cond_entropy = full_entropy - single_entropy

    wandb.log({"invariance": invariance, "full_entropy": full_entropy, "cond_entropy": cond_entropy, "single_entropy": single_entropy})

    loss = - invariance - single_entropy * entropy_multiplier 
    return loss, single_entropy * entropy_multiplier

def sphereprob_loss_func_single_gauss(
    z1: torch.Tensor, z2: torch.Tensor, gamma1: torch.Tensor, gamma2: torch.Tensor,
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
    assert not torch.isnan(z1).any() and not torch.isinf(z1).any(), "NaNs/Infs in z1"
    assert not torch.isnan(z2).any() and not torch.isinf(z2).any(), "NaNs/Infs in z2"

    eps = torch.diag(1e-6 * torch.eye(z1.size(1), device=z1.device))

    means1 = gather(z1)
    means2 = gather(z2)
    gamma1 = gather(gamma1)
    gamma2 = gather(gamma2)

    sim = sphere_similarity(means1, means2, gamma1, gamma2, prediction_precision)
    invariance = torch.mean(sim) 

    means1 = means1 - means1.mean(dim=0)
    means2 = means2 - means2.mean(dim=0)
    means = torch.cat([means1, means2], dim=0)
    N, D = means.size()
    cov = (means.T @ means) / (N - 1)
    cov1 = (means1.T @ means1) / (N//2 - 1) 
    cov2 = (means2.T @ means2) / (N//2 - 1)

    full_entropy = log_det_expansion(cov + eps) / D
    single_entropy = 0.5 * (log_det_expansion(cov1 + eps) + log_det_expansion(cov2 + eps)) / D
    cond_entropy = full_entropy - single_entropy

    wandb.log({"invariance": invariance, "full_entropy": full_entropy, "cond_entropy": cond_entropy, "single_entropy": single_entropy})


    assert not torch.isnan(full_entropy).any() and not torch.isinf(full_entropy).any(), "NaNs/Infs in z1"
    assert not torch.isnan(invariance).any() and not torch.isinf(invariance).any(), "NaNs/Infs in z2"

    loss = - invariance - entropy_multiplier * single_entropy
    return loss, entropy_multiplier * single_entropy

def sphereprob_loss_func_single_knn(
    z1: torch.Tensor, z2: torch.Tensor, gamma1: torch.Tensor, gamma2: torch.Tensor,
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
    assert not torch.isnan(z1).any() and not torch.isinf(z1).any(), "NaNs/Infs in z1"
    assert not torch.isnan(z2).any() and not torch.isinf(z2).any(), "NaNs/Infs in z2"

    means1 = gather(z1)
    means2 = gather(z2)
    gamma1 = gather(gamma1)
    gamma2 = gather(gamma2)

    sim = sphere_similarity(means1, means2, gamma1, gamma2, prediction_precision)
    invariance = torch.mean(sim) 

    full_entropy = kozachenko_leonenko_dual_entropy(means1, means2)
    single_entropy = 0.5 * (kozachenko_leonenko_single_entropy(means1) + kozachenko_leonenko_single_entropy(means2))
    cond_entropy = full_entropy - single_entropy

    wandb.log({"invariance": invariance, "full_entropy": full_entropy, "cond_entropy": cond_entropy, "single_entropy": single_entropy})


    assert not torch.isnan(full_entropy).any() and not torch.isinf(full_entropy).any(), "NaNs/Infs in z1"
    assert not torch.isnan(invariance).any() and not torch.isinf(invariance).any(), "NaNs/Infs in z2"

    loss = - invariance - entropy_multiplier * single_entropy 
    return loss, entropy_multiplier * single_entropy


def sphereprob_loss_func(
        z1: torch.Tensor, z2: torch.Tensor, recognition_precision: float = 0.2,
        prediction_precision: float = 0.2,
        type: str = "dual", entropy_multiplier: float = 1.0
) -> torch.Tensor:
    
    gammas1 = torch.ones(z1.size(0), device=z1.device) * recognition_precision
    gammas2 = torch.ones(z2.size(0), device=z2.device) * recognition_precision
    
    if type == "single_sample":
        return sphereprob_loss_func_single(z1, z2, gammas1, gammas2, prediction_precision, entropy_multiplier)
    
    elif type == "single_knn":
        return sphereprob_loss_func_single_knn(z1, z2, gammas1, gammas2, prediction_precision, entropy_multiplier)
    
    elif type == "single_gauss":
        return sphereprob_loss_func_single_gauss(z1, z2, gammas1, gammas2, prediction_precision, entropy_multiplier)

    elif type == "dual_sample":
        return sphereprob_loss_func_dual(z1, z2, gammas1, gammas2, prediction_precision)
    
    elif type == "dual_knn":
        return sphereprob_loss_func_dual_knn(z1, z2, gammas1, gammas2, prediction_precision)
    
    elif type == "dual_gauss":
        return sphereprob_loss_func_dual_gauss(z1, z2, gammas1, gammas2, prediction_precision)

    
    raise ValueError(f"Loss type {type} not supported.")