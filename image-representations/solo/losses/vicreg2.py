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
from solo.utils.misc import gather

def prior_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes prior loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: prior regularization loss.
    """
    z = torch.cat([z1, z2], dim=0)
    cov = torch.cov(z.T).double()
    lambdas = torch.linalg.eigvalsh(cov)
    target_lambdas = 1 / torch.arange(1, lambdas.size(0) + 1, device=lambdas.device, dtype=lambdas.dtype)
    loss = F.mse_loss(torch.log(lambdas), torch.log(target_lambdas))
    return loss.type(z1.dtype)

def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes mse loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: invariance loss (mean squared error).
    """

    return F.mse_loss(z1, z2)

def variance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes variance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: variance regularization loss.
    """

    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    return std_loss


def covariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: covariance regularization loss.
    """

    z = torch.cat([z1, z2], dim=0)

    N, D = z.size()

    z = z - z.mean(dim=0)
    cov_z = (z.T @ z) / (N - 1)

    diag = torch.eye(D, device=z.device)
    cov_loss = cov_z[~diag.bool()].pow_(2).sum() / D 
    return cov_loss

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
    
    

# def log_det_approx(A: torch.Tensor) -> torch.Tensor:
#     """
#     Approximates the log-determinant of a matrix A using diagonal and 
#     off-diagonal terms.
    
#     This approximation is accurate for diagonally dominant matrices.
    
#     Args:
#         A (torch.Tensor): A square 2D tensor.
        
#     Returns:
#         torch.Tensor: A scalar tensor containing the approximate log-determinant.
#     """

#     # --- 1. Diagonal Term ---
#     # Sum of the logs of the diagonal elements: sum(log(A_ii))
#     diag_A = torch.diag(A) + 1e-9  # Add a small epsilon for numerical stability
#     diagonal_term = torch.log(diag_A).sum()

#     # --- 2. Off-Diagonal Correction Term ---
#     # -0.5 * sum_{i != j} (A_ij * A_ji) / (A_ii * A_jj)
    
#     # Create a matrix of denominators (A_ii * A_jj) using an outer product
#     # Add a small epsilon for numerical stability if diagonal elements are near zero
#     eps = 1e-9
#     denominator_matrix = torch.outer(diag_A, diag_A) + eps
    
#     # Create the matrix of numerators (A_ij * A_ji)
#     # This is an element-wise product of A and its transpose
#     numerator_matrix = A * A.T
    
#     # Calculate the full fraction matrix
#     fraction_matrix = numerator_matrix / denominator_matrix
    
#     # To sum only the off-diagonal elements (where i != j), we can sum the
#     # whole matrix and subtract the sum of the diagonal.
#     off_diagonal_sum = fraction_matrix.sum() - torch.diag(fraction_matrix).sum()
    
#     off_diagonal_term = -0.5 * off_diagonal_sum

#     return diagonal_term + off_diagonal_term

def vicreg2_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Computes VICReg's loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        sim_loss_weight (float): invariance loss weight.
        var_loss_weight (float): variance loss weight.
        cov_loss_weight (float): covariance loss weight.

    Returns:
        torch.Tensor: VICReg loss.
    """

    sim_loss = invariance_loss(z1, z2) 

    # vicreg's official code gathers the tensors here
    # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    z1, z2 = gather(z1), gather(z2)


    # # approximation
    # z = torch.cat([z1, z2], dim=0)
    # z = z - z.mean(dim=0)
    # N, D = z.size()
    # cov_z = (z.T @ z) / (N - 1)
    # neg_entropy = - log_det_expansion(cov_z) / D
    # loss = sim_loss_weight * sim_loss + cov_loss_weight * neg_entropy / 2

    ## vicreg loss
    var_loss = variance_loss(z1, z2)
    cov_loss = covariance_loss(z1, z2)
    loss = sim_loss_weight * sim_loss + var_loss_weight * var_loss + cov_loss_weight * cov_loss
    entropy = var_loss_weight * var_loss + cov_loss_weight * cov_loss

    ## exact logdet
    # cov_z = (z.T @ z) / (N - 1) + 1e-6 * torch.eye(D, device=z.device)
    # cov_loss = - torch.logdet(cov_z) / D
    # loss = sim_loss + cov_loss
    


    return loss, entropy
