
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from probssl.losses.ssl.qalmanSSL import qalmanSSL_loss_func_stopgrad, qalmanSSL_loss_func_knn, qalmanSSL_loss_func_logdet, qalmanSSL_loss_func_kde
from probssl.losses.ssl.entropy import gauss_entropy_loss
from probssl.methods.base import BaseMethod

from probssl.utils.misc import omegaconf_select
from probssl.backbones.mlp import MLP
from probssl.backbones.variance_mlp import VarianceMLP

_RNN_BACKBONES = {
    "rnn": nn.RNN,
    "lstm": nn.LSTM,
    "gru": nn.GRU,
}

_VARIANCE_DEPENDENCY = ["state", "observation", None]

_LOSS_TYPES = ["stopgrad", "knn", "logdet", "kde"]

class QalmanSSLEncoder(nn.Module):

    def __init__(self, input_dim, state_dim, 
                 encoder_hidden_dim=100, encoder_hidden_layers=1, rnn_hidden_dim=10, rnn_num_layers=1, 
                 rnn_backbone=nn.LSTM, variance_dependency="state",):
        super(QalmanSSLEncoder, self).__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.variance_dependency = variance_dependency

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=state_dim,
            n_hidden_layers=encoder_hidden_layers,
            # activation=nn.GELU
        )

        self.rnn = rnn_backbone(
            input_size=state_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )

        self.predictor = MLP(
            input_dim=rnn_hidden_dim,
            hidden_dim=20,
            output_dim=state_dim,
            n_hidden_layers=2,
            # activation=nn.GELU
        )

        if self.variance_dependency == "state":
            self.variance_predictor = MLP(
                input_dim=rnn_hidden_dim,
                hidden_dim=state_dim, # not used
                output_dim=state_dim,
                n_hidden_layers=0,
                output_activation=nn.Softplus
            )
        elif self.variance_dependency == "observation":
            self.variance_predictor = VarianceMLP(
                input_dim=input_dim,
                hidden_dim=input_dim, 
                output_dim=state_dim,
                output_activation=None # positivity is ensured in the VarianceMLP
            )


    def forward(self, observations):
        inferences = self.encoder(observations)
        estimates, hidden_states = self.rnn(inferences)
        predictions = self.predictor(estimates)

        if isinstance(hidden_states, tuple):  # LSTM
            hidden_states = hidden_states[0]  # take only the hidden state, ignore cell

        if self.variance_dependency == "state":
            shifted_estimates = torch.roll(estimates, shifts=1, dims=1)
            shifted_estimates[:, 0, :] = 0
            predictions_variances = self.variance_predictor(shifted_estimates)
        elif self.variance_dependency == "observation":
            predictions_variances = self.variance_predictor(observations)
        elif self.variance_dependency is None:
            # set variances to one
            predictions_variances = torch.ones_like(predictions)

        states = {
            "inferences": inferences,
            "estimates": estimates,
            "predictions": predictions,
            "predictions_variances": predictions_variances,
            "hidden_states": hidden_states,
        }

        return states

class QalmanSSL(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements QalmanSSL 
        """

        super().__init__(cfg)
        # This runs
        # cfg = self.add_and_assert_specific_cfg(cfg)
        # self.cfg = cfg

        self.input_dim = cfg.method_kwargs.input_dim
        self.state_dim = cfg.method_kwargs.state_dim
        self.rnn_hidden_dim = cfg.method_kwargs.rnn_hidden_dim
        self.encoder_hidden_dim = cfg.method_kwargs.encoder_hidden_dim
        self.rnn_num_layers = cfg.method_kwargs.rnn_num_layers
        self.rnn_backbone = _RNN_BACKBONES[cfg.method_kwargs.rnn_backbone]
        self.loss_type = cfg.method_kwargs.loss_type
        self.variance_dependency = cfg.method_kwargs.variance_dependency

        self.encoder = QalmanSSLEncoder(
            input_dim=self.input_dim,
            state_dim=self.state_dim,
            encoder_hidden_dim=self.encoder_hidden_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            rnn_num_layers=self.rnn_num_layers,
            rnn_backbone=self.rnn_backbone,
            variance_dependency=self.variance_dependency,
        )

        # learning rate multipliers
        self.encoder_lr_multiplier = cfg.method_kwargs.encoder_lr_multiplier
        self.rnn_lr_multiplier = cfg.method_kwargs.rnn_lr_multiplier
        self.predictor_lr_multiplier = cfg.method_kwargs.predictor_lr_multiplier
        self.variance_predictor_lr_multiplier = cfg.method_kwargs.variance_predictor_lr_multiplier

        self.inference_entropy_loss_multiplier = cfg.method_kwargs.inference_entropy_loss_multiplier


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        # Make sure to call the parent method first
        cfg = BaseMethod.add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.input_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.state_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.rnn_hidden_dim")
        cfg.method_kwargs.encoder_hidden_dim = omegaconf_select(cfg, "method_kwargs.encoder_hidden_dim", 100)
        cfg.method_kwargs.encoder_hidden_layers = omegaconf_select(cfg, "method_kwargs.encoder_hidden_layers", 1)
        cfg.method_kwargs.rnn_num_layers = omegaconf_select(cfg, "method_kwargs.rnn_num_layers", 1)

        cfg.method_kwargs.encoder_lr_multiplier = omegaconf_select(cfg, "method_kwargs.encoder_lr_multiplier", 2.0)
        cfg.method_kwargs.rnn_lr_multiplier = omegaconf_select(cfg, "method_kwargs.rnn_lr_multiplier", 20.0)
        cfg.method_kwargs.predictor_lr_multiplier = omegaconf_select(cfg, "method_kwargs.predictor_lr_multiplier", 30.0)
        cfg.method_kwargs.variance_predictor_lr_multiplier = omegaconf_select(cfg, "method_kwargs.variance_predictor_lr_multiplier", 20.0)

        cfg.method_kwargs.inference_entropy_loss_multiplier = omegaconf_select(cfg, "method_kwargs.inference_entropy_loss_multiplier", 0.0)

        cfg.method_kwargs.loss_type = omegaconf_select(cfg, "method_kwargs.loss_type", "stopgrad")
        assert cfg.method_kwargs.loss_type in _LOSS_TYPES, f"Choose from {_LOSS_TYPES}"

        cfg.method_kwargs.variance_dependency = omegaconf_select(cfg, "method_kwargs.variance_dependency", "state")
        assert cfg.method_kwargs.variance_dependency in _VARIANCE_DEPENDENCY, f"Choose from {_VARIANCE_DEPENDENCY}"

        cfg.method_kwargs.rnn_backbone = omegaconf_select(cfg, "method_kwargs.rnn_backbone", "lstm")
        assert cfg.method_kwargs.rnn_backbone in _RNN_BACKBONES, f"Choose from {_RNN_BACKBONES.keys()}"

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        qf = self.encoder
        extra_learnable_params = [
            {'params': qf.encoder.parameters(), 'lr': self.encoder_lr_multiplier * self.lr},
            {'params': qf.rnn.parameters(), 'lr': self.rnn_lr_multiplier * self.lr},
            {'params': qf.predictor.parameters(), 'lr': self.predictor_lr_multiplier * self.lr},
        ]
        if self.variance_dependency is not None:
            extra_learnable_params.append(
                {'params': qf.variance_predictor.parameters(), 'lr': self.variance_predictor_lr_multiplier * self.lr}
            )
        return super().learnable_params + extra_learnable_params

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, X, Y], where
                img_indexes (torch.Tensor): indexes of the images in the batch.
                X (torch.Tensor): input data.
                Y (torch.Tensor): labels of the input data.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss 
        """
        out = super().training_step(batch, batch_idx)

        decoder_loss = out["decoder_loss"]
        latents = out["latents"]
        zs_inf = latents["inferences"]
        zs_pred = latents["predictions"]
        zs_pred_variances = latents["predictions_variances"]

        if self.loss_type == "stopgrad":
            ssl_loss = qalmanSSL_loss_func_stopgrad(
                zs_inf,
                zs_pred,
                zs_pred_variances,
            )
        elif self.loss_type == "knn":
            ssl_loss = qalmanSSL_loss_func_knn(
                zs_inf,
                zs_pred,
                zs_pred_variances,
            )
        elif self.loss_type == "logdet":
            ssl_loss = qalmanSSL_loss_func_logdet(
                zs_inf,
                zs_pred,
                zs_pred_variances,
            )
        elif self.loss_type == "kde":
            ssl_loss = qalmanSSL_loss_func_kde(
                zs_inf,
                zs_pred,
                zs_pred_variances,
            )

        # entropy_loss = gauss_entropy_loss(zs_inf.view(-1, zs_inf.shape[-1]))
        # ssl_loss += self.inference_entropy_loss_multiplier * entropy_loss

        loss = ssl_loss + decoder_loss

        metrics = {
            "ssl_loss": ssl_loss,
            "loss": loss,
            "predictions_variances": zs_pred_variances.mean(),
        }
        self.log_dict(metrics, on_epoch=None)

        return loss


