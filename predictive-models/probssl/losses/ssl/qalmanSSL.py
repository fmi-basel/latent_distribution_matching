

import torch
import torch.nn.functional as F
import numpy as np
from probssl.utils.metrics import gauss_cross_entropy

def qalmanSSL_loss_func_stopgrad(
            zs_inf: torch.Tensor,
            zs_pred: torch.Tensor,
            zs_pred_variances: torch.Tensor) -> torch.Tensor:
    
    batch_size, n_steps, state_dim = zs_inf.shape

    zs_pred_covariances = torch.diag_embed(zs_pred_variances)

    alignment_sum = 0.0
    for t in range(1,n_steps):
        z_inf = zs_inf[:,t]
        z_pred = zs_pred[:,t-1]

        z_pseudo = z_inf.detach()
        Sigma_e = zs_pred_covariances[:,t]

        alignment = - gauss_cross_entropy(z_pred, z_pseudo, Sigma_e)
        alignment_sum += alignment

    loss = - alignment_sum.mean() 
    return loss

def qalmanSSL_loss_func_knn(
            zs_inf: torch.Tensor,
            zs_pred: torch.Tensor,
            zs_pred_variances: torch.Tensor) -> torch.Tensor:
    
    batch_size, n_steps, state_dim = zs_inf.shape

    zs_pred_covariances = torch.diag_embed(zs_pred_variances)

    alignment_sum = 0.0
    for t in range(1,n_steps):
        z_inf = zs_inf[:,t]
        z_pred = zs_pred[:,t-1]

        z_pseudo = z_inf
        Sigma_e = zs_pred_covariances[:,t]

        alignment = - gauss_cross_entropy(z_pred, z_pseudo, Sigma_e)
        alignment_sum += alignment

    # compute average entropy of inferred states
    random_indices = np.random.choice(batch_size * n_steps, size=batch_size * 5, replace=False) # sample to reduce computation
    zs_inf_random = zs_inf.view(batch_size * n_steps, state_dim)[random_indices]
    entropy = kozachenko_leonenko_single_entropy(zs_inf_random, k=3, p=2, eps=1e-8, last_only=True)

    loss = - alignment_sum.mean() - entropy * n_steps
    return loss


def qalmanSSL_loss_func_logdet(
            zs_inf: torch.Tensor,
            zs_pred: torch.Tensor,
            zs_pred_variances: torch.Tensor) -> torch.Tensor:
    
    batch_size, n_steps, state_dim = zs_inf.shape

    zs_pred_covariances = torch.diag_embed(zs_pred_variances)

    alignment_sum = 0.0
    for t in range(1,n_steps):
        z_inf = zs_inf[:,t]
        z_pred = zs_pred[:,t-1]

        z_pseudo = z_inf
        Sigma_e = zs_pred_covariances[:,t]

        alignment = - gauss_cross_entropy(z_pred, z_pseudo, Sigma_e)
        alignment_sum += alignment

    entropy = logdet_entropy(zs_inf.view(batch_size * n_steps, state_dim), eps=1e-8)

    loss = - alignment_sum.mean() - entropy * n_steps
    return loss


def qalmanSSL_loss_func_kde(
            zs_inf: torch.Tensor,
            zs_pred: torch.Tensor,
            zs_pred_variances: torch.Tensor) -> torch.Tensor:
    
    batch_size, n_steps, state_dim = zs_inf.shape

    bandwidth = torch.mean(torch.sqrt(zs_pred_variances)).item()
    zs_pred_covariances = torch.diag_embed(zs_pred_variances)

    alignment_sum = 0.0
    for t in range(1,n_steps):
        z_inf = zs_inf[:,t]
        z_pred = zs_pred[:,t-1]

        z_pseudo = z_inf
        Sigma_e = zs_pred_covariances[:,t]

        alignment = - gauss_cross_entropy(z_pred, z_pseudo, Sigma_e)
        alignment_sum += alignment
        
    entropy = kde_entropy(zs_inf.view(batch_size * n_steps, state_dim), bandwidth=bandwidth, eps=1e-8)

    loss = - alignment_sum.mean() - entropy * n_steps
    return loss

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

def logdet_entropy(x, eps=1e-8):
    """Entropy estimator for batch x~p(x).
        :param x: prediction; shape [T, N]
        :return: scalar
        """
    if type(x) is np.ndarray:
        x = torch.tensor(x.astype(np.float32))

    cov = torch.cov(x.T)  # shape [N, N]
    cov += eps * torch.eye(cov.shape[0], device=cov.device)  # numerical stability

    sign, logdet = torch.slogdet(cov)
    d = x.shape[1]

    ent = 0.5 * logdet + 0.5 * d * (1 + np.log(2 * np.pi))

    return ent

def kde_entropy(x, bandwidth=0.1, eps=1e-8):
    """Entropy estimator for batch x~p(x) via KDE.
        :param x: prediction; shape [T, N]
        :return: scalar
        """
    if type(x) is np.ndarray:
        x = torch.tensor(x.astype(np.float32))

    n_samples, n_dim = x.shape

    # compute pairwise squared distances
    distmat = torch.cdist(x, x, p=2)  # shape [T, T]
    distmat_squared = distmat ** 2

    # compute kernel values
    kernel_vals = torch.exp(-distmat_squared / (2 * bandwidth ** 2))  # shape [T, T]

    # compute densities
    densities = kernel_vals.sum(dim=1) / (n_samples * (bandwidth ** n_dim) * ((2 * np.pi) ** (n_dim / 2)))  # shape [T]

    densities = torch.clip(densities, min=eps)  # avoid log(0)

    ent = -torch.log(densities).mean()

    return ent