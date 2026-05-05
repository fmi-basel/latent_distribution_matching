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
import numpy as np

def simsiam_loss_func(z1: torch.Tensor, z2: torch.Tensor, predictor) -> torch.Tensor:
    """Computes SimSiam's loss given batch of predicted features p from view 1 and
    a batch of projected features z from view 2.

    Args:
        p (torch.Tensor): Tensor containing predicted features from view 1.
        z (torch.Tensor): Tensor containing projected features from view 2.
        simplified (bool): faster computation, but with same result.

    Returns:
        torch.Tensor: SimSiam loss.
    """

    p1 = predictor(z1)
    p2 = predictor(z2)
    p1_sg = predictor(z1.detach())  # stop-gradient
    p2_sg = predictor(z2.detach())  # stop-gradient

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    p1 = F.normalize(p1, dim=-1)
    p2 = F.normalize(p2, dim=-1)
    p1_sg = F.normalize(p1_sg, dim=-1)
    p2_sg = F.normalize(p2_sg, dim=-1)

    goal = (p1 * z2).sum(dim=1).mean() / 2 + (p2 * z1).sum(dim=1).mean() / 2
    goal += - (p1.detach() * z2).sum(dim=1).mean() / 2 - (p2.detach() * z1).sum(dim=1).mean() / 2
    # linear_predictor = 
    # goal += (p1_sg * z2.detach()).sum(dim=1).mean() / 2 + (p2_sg * z1.detach()).sum(dim=1).mean() / 2

    # goal = (p1 * z2.detach()).sum(dim=1).mean() / 2 + (p2 * z1.detach()).sum(dim=1).mean() / 2
    # entropy = kozachenko_leonenko_single_entropy(z1, k=3, p=2) / 2 + kozachenko_leonenko_single_entropy(z2, k=3, p=2) / 2
    # goal += entropy

    return - 0.5 * goal



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

def kozachenko_leonenko_single_entropy(x, k=3, p=2, eps=1e-8):
    """Entropy estimator for batch x~p(x).
        :param x: prediction; shape [T, N]
        :param k:
        :return: scalar
        """
    if type(x) is np.ndarray:
        x = torch.tensor(x.astype(np.float32))

    nns_xx = knn(x, k=k, p=p, last_only=True)
    nns_xx = nns_xx - torch.clip(nns_xx, max=0.0) # ensure non-negative
    nns_xx = torch.clip(nns_xx, max=1e6) # avoid inf

    # deal with outliers
    # these representations are already well separated, so clip extreme distances
    upper_bound = torch.quantile(nns_xx, 0.9).detach()
    nns_xx = torch.clip(nns_xx, max=upper_bound)

    ent = torch.log(nns_xx + eps).mean() 

    return ent