

import torch
import torch.nn.functional as F
from probssl.utils.metrics import gauss_cross_entropy

def kalmanSSL_loss_func(
            zs_inf: torch.Tensor,
            zs_pred: torch.Tensor,
            zs_inf_covariances: torch.Tensor,
            zs_pred_covariances: torch.Tensor) -> torch.Tensor:
    
    batch_size, n_steps, state_dim = zs_inf.shape
    
    alignment_sum = 0.0
    for t in range(n_steps):
        z_inf = zs_inf[:,t]
        z_pred = zs_pred[:,t]

        Sigma_D = zs_inf_covariances[:,t]
        Sigma_z_pred = zs_pred_covariances[:,t]

        Sigma_e = Sigma_z_pred + Sigma_D
        alignment = - gauss_cross_entropy(z_pred, z_inf.detach(), Sigma_e).mean()
        alignment_sum += alignment

    loss = - alignment_sum
    return loss
