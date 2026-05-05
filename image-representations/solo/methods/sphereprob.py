
from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.sphereprob import sphereprob_loss_func
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
    elif loss_type == "single_gauss":
        return "dual_gauss"
    elif loss_type == "dual_gauss":
        return "single_gauss"

class SphereProb(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

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
        self.recognition_precision: float = cfg.method_kwargs.recognition_precision

        self.use_predictor: bool = cfg.method_kwargs.use_predictor

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        if self.use_predictor:
            self.predictor = nn.Linear(proj_output_dim, proj_output_dim, bias=False)
        else:
            self.predictor = nn.Identity()

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SphereProb, SphereProb).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.prediction_precision")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.recognition_precision")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.loss.type")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.loss.entropy_multiplier")
        
        cfg.method_kwargs.use_predictor = omegaconf_select(
            cfg, "method_kwargs.predictor", default=False
        )

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters(), "lr": self.lr},
                                  {"name": "predictor", "params": self.predictor.parameters(), "lr": 5*self.lr}]
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
        # depending on if we want to use the length as variance or not
        z = F.normalize(z, dim=-1)
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
        z1 = self.predictor(z1)

        nce_loss, entropy = sphereprob_loss_func(
            z1, z2,
            recognition_precision=self.recognition_precision,
            prediction_precision=self.prediction_precision,
            type=self.loss_type,
            entropy_multiplier=self.entropy_multiplier
        )

        _, entropy2 = sphereprob_loss_func(
            z1, z2,
            recognition_precision=self.recognition_precision,
            prediction_precision=self.prediction_precision,
            type=invert(self.loss_type),
            entropy_multiplier=self.entropy_multiplier
        )

        cos_sim = compute_gradient_cos_similarity(entropy, entropy2, self)
        self.log("train_entropy_cos_sim", cos_sim, on_epoch=True, sync_dist=True)
        
        if not torch.isfinite(nce_loss + class_loss):
            print("Non-finite loss:", nce_loss)
            print("Non-finite loss:", class_loss)
            raise RuntimeError("Non-finite loss detected")

        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             if torch.isnan(param.grad).any():
    #                 self.print(f"Warning: Gradient for {name} contains NaN!")
    #             if torch.isinf(param.grad).any():
    #                 self.print(f"Warning: Gradient for {name} contains Inf!")