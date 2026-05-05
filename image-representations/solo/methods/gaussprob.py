

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from solo.losses.gaussprob import gaussprob_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select, compute_gradient_cos_similarity


def invert(loss_type: str) -> str:
    if loss_type == "single_sample":
        return "dual_sample"
    elif loss_type == "dual_sample":
        return "single_sample"
    elif loss_type == "single_knn":
        return "dual_knn"
    elif loss_type == "dual_knn":
        return "single_knn"

class GaussProb(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """
        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        self.loss_type: str = cfg.method_kwargs.loss.type
        self.entropy_multiplier: float = cfg.method_kwargs.loss.entropy_multiplier

        self.prediction_precision: float = cfg.method_kwargs.prediction_precision
        self.kernel_precision: float = cfg.method_kwargs.kernel_precision

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(GaussProb, GaussProb).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.prediction_precision")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.kernel_precision")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.loss.type")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.loss.entropy_multiplier")


        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        if self.loss_type == "causal":
            extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()},
                                    {"name": "predictor", 
                                    "params": self.predictor.parameters(),
                                    "lr": self.lr * 10,
                                    "weight_decay": 0}]
        else:
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
        # vars = self.variance_projector(out["feats"])
        out.update({"z": z, "vars": vars})
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

        nce_loss, entropy = gaussprob_loss_func(
            z1, z2,
            prediction_precision=self.prediction_precision,
            kernel_precision=self.kernel_precision,
            type=self.loss_type,
            entropy_multiplier=self.entropy_multiplier
        )

        _, entropy2 = gaussprob_loss_func(
            z1, z2,
            prediction_precision=self.prediction_precision,
            kernel_precision=self.kernel_precision,
            type=invert(self.loss_type),
            entropy_multiplier=self.entropy_multiplier
        )

        cos_sim = compute_gradient_cos_similarity(entropy, entropy2, self)
        self.log("train_entropy_cos_sim", cos_sim, on_epoch=True, sync_dist=True)


        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss
