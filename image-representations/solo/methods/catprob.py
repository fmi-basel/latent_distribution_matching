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

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import numpy as np
from solo.losses.catprob import catprob_loss_func
from solo.methods.base import BaseMethod

import torch.nn.utils.parametrizations as param
import torch.nn.functional as F

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)
    
class CatProb(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        warmup_match_percentage : float = cfg.method_kwargs.warmup_match_percentage
        end_match_percentage : float = cfg.method_kwargs.end_match_percentage

        self.start_temperature: float = 1 / np.log((proj_output_dim - 1) * warmup_match_percentage / (1 - warmup_match_percentage))
        self.temperature: float = self.start_temperature
        self.end_temperature: float = 1 / np.log((proj_output_dim - 1) * end_match_percentage / (1 - end_match_percentage))
        # self.predictor_regularization: float = cfg.method_kwargs.predictor_regularization


        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.Softmax(dim=-1),
            # L2NormalizationLayer(dim=-1)
        )

        self.loss_type = cfg.method_kwargs.loss.type
        self.entropy_multiplier = cfg.method_kwargs.loss.entropy_multiplier

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(CatProb, CatProb).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.loss.type")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.loss.entropy_multiplier")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.warmup_match_percentage")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.end_match_percentage")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out


    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        z1, z2 = out["z"]
        cat_loss = catprob_loss_func(
            z1, z2,
            temperature=self.temperature,
            end_temperature=self.end_temperature,
            type=self.loss_type,
            entropy_multiplier=self.entropy_multiplier
        )

        self.log("train_cat_loss", cat_loss, on_epoch=True, sync_dist=True)

        return cat_loss + class_loss

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        # exponential annealing temperature schedule
        if self.current_epoch < self.max_epochs:
            # self.temperature = self.end_temperature + (self.start_temperature - self.end_temperature) * 0.5 * (1 + torch.cos(torch.tensor(self.current_epoch / self.max_epochs * 3.1415926535)))
            # self.temperature = self.start_temperature * (self.end_temperature / self.start_temperature) ** ((self.current_epoch / self.max_epochs) ** 0.5)
            self.temperature = self.temperature - 4 / self.max_epochs * (self.temperature - self.end_temperature) 
        self.log("temperature", self.temperature)