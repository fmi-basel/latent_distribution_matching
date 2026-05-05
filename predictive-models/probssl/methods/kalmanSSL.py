
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from probssl.losses.ssl.kalmanSSL import kalmanSSL_loss_func
from probssl.losses.ssl.procrusteSSL import procusteSSL_loss_func, stopgrad_entropy_term, kl_entropy_term
from probssl.methods.base import BaseMethod

from probssl.utils.misc import omegaconf_select
from probssl.backbones.mlp import MLP
from probssl.backbones.variance_mlp import VarianceMLP
from probssl.utils.misc import get_padded_view

_SIGMA_A_DEPENDENCY = ["error", "observation+prediction", None]
_SIGMA_D_DEPENDENCY = ["state", "observation", "observation-var", None]
_LOSS_FUNC_TYPES = ["regular", "procrustes"]

class KalmanSSLEncoder(nn.Module):

    def __init__(self, input_dim, latent_dim, state_dim, hidden_dim=100, sigma_D_dependency=None, sigma_A_dependency=None, sigma_A_pred_steps=10):
        super(KalmanSSLEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.sigma_D_dependency = sigma_D_dependency
        self.sigma_A_dependency = sigma_A_dependency
        
        # inference
        self.F_inf = MLP(input_dim, hidden_dim, latent_dim)
        self.D = nn.Parameter(torch.zeros(latent_dim, state_dim, dtype=torch.float32), requires_grad=False)
        # set D to eye for latent 
        dim = min(latent_dim, state_dim)
        self.D[:dim, :dim] = torch.eye(dim, dtype=torch.float32)

        self.joint_sigma_scaling = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        if sigma_D_dependency is None:
            self.log_Sigma_D_diag = nn.Parameter(torch.zeros(latent_dim), requires_grad=False)
        elif sigma_D_dependency == "observation-var":
            self.Sigma_D_diag_function = VarianceMLP(input_dim, input_dim, latent_dim, output_activation=nn.ReLU)
        elif sigma_D_dependency == "observation":
            act = lambda: nn.Softplus(beta=10.0)
            self.Sigma_D_diag_function = MLP(input_dim, hidden_dim, latent_dim, n_hidden_layers=2, activation=nn.ReLU, output_activation=act)
        elif sigma_D_dependency == "state":
            self.log_Sigma_D_diag_function = MLP(state_dim, hidden_dim, latent_dim)

        # State transition 
        self.A = nn.Parameter(torch.eye(state_dim) + torch.randn(state_dim, state_dim) * 0.1, requires_grad=True)
        # rescale A to have spectral radius < 1
        with torch.no_grad():
            spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(self.A)))
            self.A /= spectral_radius * 1.1
        self.b_z = nn.Parameter(torch.zeros(state_dim))
        # Covariance of state prediction
        self.log_Sigma_A_diag = nn.Parameter(torch.zeros(state_dim)) 
        if sigma_A_dependency == "error":
            self.sigma_A_pred_steps = sigma_A_pred_steps
            # act = lambda: nn.Softplus(beta=10.0)
            act = nn.Sigmoid
            self.Sigma_A_diag_function = MLP(latent_dim*sigma_A_pred_steps, 20, state_dim, 
                                             n_hidden_layers=1, activation=nn.ReLU, output_activation=act)
        elif sigma_A_dependency == "observation+prediction":
            self.sigma_A_pred_steps = sigma_A_pred_steps
            # act = lambda: nn.Softplus(beta=10.0)
            act = nn.Sigmoid
            self.Sigma_A_diag_function = MLP(sigma_A_pred_steps*2*latent_dim, 20, state_dim, 
                                             n_hidden_layers=1, activation=nn.ReLU, output_activation=act)

        # Parameter learning rates
        self.super_slow_params = []
        self.slow_params = ["F_inf",]
        self.normal_params = ["Sigma_A_diag_function"]
        self.fast_params = ["A", "b_z", "joint_sigma_scaling", "log_Sigma_A_diag"]
        self.super_fast_params = []
        if sigma_D_dependency == "observation-var":
            self.super_slow_params.append("Sigma_D_diag_function")
        elif sigma_D_dependency == "observation":
            self.super_slow_params.append("Sigma_D_diag_function")
        elif sigma_D_dependency == "state":
            self.slow_params.append("log_Sigma_D_diag_function")


    def forward(self, observations):
        batch_size = observations.shape[0]
        n_steps = observations.shape[1]
        state_dim = self.state_dim
        latent_dim = self.latent_dim
        input_dim = self.input_dim
        
        # Initialize state
        z_est = torch.zeros(batch_size, state_dim, dtype=observations.dtype, device=observations.device)
        Sigma_z_est = torch.eye(state_dim, dtype=observations.dtype, device=observations.device).repeat(batch_size, 1, 1)
        eye_batched = torch.eye(state_dim, dtype=observations.dtype, device=observations.device).repeat(batch_size, 1, 1)

        # inference of pseudo observation
        zs_inf = self.F_inf(observations) 

        # allocate state variables
        inferences = zs_inf
        inferences_covariances = torch.zeros((batch_size, n_steps, latent_dim, latent_dim), dtype=zs_inf.dtype, device=zs_inf.device)
        estimates = torch.zeros((batch_size, n_steps, state_dim), dtype=zs_inf.dtype, device=zs_inf.device)
        estimates_covariances = torch.zeros((batch_size, n_steps, state_dim, state_dim), dtype=zs_inf.dtype, device=zs_inf.device)
        predictions = torch.zeros((batch_size, n_steps, latent_dim), dtype=zs_inf.dtype, device=zs_inf.device)
        prediction_covariances = torch.zeros((batch_size, n_steps, latent_dim, latent_dim), dtype=zs_inf.dtype, device=zs_inf.device)
        e = torch.ones((batch_size, latent_dim), dtype=zs_inf.dtype, device=zs_inf.device) # prediction error
        if self.sigma_A_dependency == "error":
            state_prediction_errors = torch.zeros((batch_size, n_steps + self.sigma_A_pred_steps, latent_dim), dtype=zs_inf.dtype, device=zs_inf.device)
        elif self.sigma_A_dependency == "observation+prediction":
            state_prediction_vars = torch.zeros((batch_size, n_steps + self.sigma_A_pred_steps, 2 * latent_dim), dtype=zs_inf.dtype, device=zs_inf.device)

        # rescale A if instability is detected
        with torch.no_grad():
            spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(self.A)))
            if spectral_radius > 1.02:
                self.A /= spectral_radius / 1.02
                print(f"Rescaled A to have spectral radius < 1: {spectral_radius:.2f} -> {torch.max(torch.abs(torch.linalg.eigvals(self.A))):.2f}")

        A_batched = self.A.unsqueeze(0).expand(batch_size, -1, -1) # [B,D,D]
        D_batched = self.D.unsqueeze(0).expand(batch_size, -1, -1) # [B,L,D]
        b_z_batched = self.b_z.unsqueeze(0).expand(batch_size, -1) # [B,D]
        
        for t in range(n_steps):
            # prediction
            # z_pred = (self.A @ (z_est - self.b_z)) + self.b_z
            z_pred = torch.einsum('bij,bj->bi', A_batched, z_est - b_z_batched) + b_z_batched # [B,D]
            z_inf_pred = torch.einsum('bij,bj->bi', D_batched, z_pred) # [B,L]

            # inference view
            z_inf = zs_inf[:,t] # inference of pseudo observation, [B,L]

            # prediction error
            e = z_inf - z_inf_pred  # [B,L]

            # get covariances
            log_Sigma_A_diag = self.log_Sigma_A_diag.unsqueeze(0).expand(batch_size, -1)
            Sigma_A = torch.diag_embed(torch.exp(log_Sigma_A_diag))
            if self.sigma_A_dependency == "error":
                # store prediction errors and estimate Sigma_A based on the last sigma_A_pred_steps steps
                state_prediction_errors[:,t-1+self.sigma_A_pred_steps] = e.detach()
                error_view = state_prediction_errors[:,t:t+self.sigma_A_pred_steps].clone() # annoyingly, we have to clone to prevent gradient issues
                Sigma_A_diag = 10 * self.Sigma_A_diag_function(error_view.view(batch_size, -1)) 
                Sigma_A = torch.diag_embed(Sigma_A_diag) + Sigma_A
                # print(Sigma_A_diag)
            elif self.sigma_A_dependency == "observation+prediction":
                # store variables and estimate Sigma_A based on the last sigma_A_pred_steps steps
                state_prediction_vars[:,t-1+self.sigma_A_pred_steps] = torch.cat((z_inf, z_inf_pred), dim=-1).detach()
                error_view = state_prediction_vars[:,t:t+self.sigma_A_pred_steps].clone()
                Sigma_A_diag = 10 * self.Sigma_A_diag_function(error_view.view(batch_size, -1))
                Sigma_A = torch.diag_embed(Sigma_A_diag) + Sigma_A
            Sigma_A = torch.exp(self.joint_sigma_scaling) * Sigma_A

            if self.sigma_D_dependency is None:
                log_Sigma_D_diag = self.log_Sigma_D_diag.unsqueeze(0).expand(batch_size, -1)
                Sigma_D = torch.diag_embed(torch.exp(log_Sigma_D_diag))
            elif self.sigma_D_dependency == "observation-var" or self.sigma_D_dependency == "observation":
                Sigma_D_diag = self.Sigma_D_diag_function(observations[:,t])
                Sigma_D = torch.diag_embed(Sigma_D_diag + 1e-5)
            elif self.sigma_D_dependency == "state":
                log_Sigma_D_diag = self.log_Sigma_D_diag_function(z_est)
                Sigma_D = torch.diag_embed(torch.exp(log_Sigma_D_diag))
            Sigma_D = torch.exp(self.joint_sigma_scaling) * Sigma_D
            
            # Prediction covariances
            # Sigma_z_pred = self.A @ Sigma_z_est @ self.A.T + Sigma_A
            Sigma_z_pred = torch.einsum('bij,bjk->bik', A_batched, torch.einsum('bij,bkj->bik', Sigma_z_est, A_batched)) + Sigma_A # [B,D,D]
            Sigma_z_inf_pred = torch.einsum('bij,bjk->bik', D_batched, torch.einsum('bij,bkj->bik', Sigma_z_pred, D_batched)) # [B,L,L]

            # Update estimate state and covariance
            Sigma_e = Sigma_D + Sigma_z_inf_pred  # [B,L,L]
            Sigma_e_inverse = torch.inverse(Sigma_e)
            # K = Sigma_z_pred @ Sigma_e_inverse  # Kalman gain
            K = torch.einsum('bij,bjk->bik', Sigma_z_pred, torch.einsum('bji,bjk->bik', D_batched, Sigma_e_inverse))  # [B,D,L]
            # z_est = z_pred + K @ e
            z_est = z_pred + torch.einsum('bjk,bk->bj', K, e) # [B,D]
            # Sigma_z_est = (eye_batched - K) @ Sigma_z_pred
            KD = torch.einsum('bij,bjk->bik', K, D_batched)  # [B,D,D]
            Sigma_z_est = torch.einsum('bij,bjk->bik', (eye_batched - KD), Sigma_z_pred)

            # Save values
            inferences[:,t] = z_inf.clone()
            inferences_covariances[:,t] = Sigma_D.clone()
            predictions[:,t] = z_inf_pred.clone()
            prediction_covariances[:,t] = Sigma_z_inf_pred.clone()
            estimates[:,t] = z_est.clone()
            estimates_covariances[:,t] = Sigma_z_est.clone()

        states = {
            "inferences": inferences,
            "inferences_covariances": inferences_covariances,
            "predictions": predictions,
            "prediction_covariances": prediction_covariances,
            "estimates": estimates,
            "estimates_covariances": estimates_covariances,
        }

        return states

class KalmanSSL(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements KalmanSSL 
        """

        super().__init__(cfg)
        # This runs
        # cfg = self.add_and_assert_specific_cfg(cfg)
        # self.cfg = cfg

        self.input_dim = cfg.method_kwargs.input_dim
        self.latent_dim = cfg.method_kwargs.latent_dim
        self.state_dim = cfg.method_kwargs.state_dim
        self.hidden_dim = cfg.method_kwargs.hidden_dim
        self.sigma_D_dependency = cfg.method_kwargs.sigma_D_dependency
        self.sigma_A_dependency = cfg.method_kwargs.sigma_A_dependency
        self.sigma_A_pred_steps = omegaconf_select(
            cfg,
            "method_kwargs.sigma_A_pred_steps",
            10,
        )

        self.encoder = KalmanSSLEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            sigma_D_dependency=self.sigma_D_dependency,
            sigma_A_dependency=self.sigma_A_dependency,
            sigma_A_pred_steps=self.sigma_A_pred_steps,
        )

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

        # size of the state space (dynamics)
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.state_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.input_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.hidden_dim")

        # size of the latent space (pseudo observation)
        cfg.method_kwargs.latent_dim = omegaconf_select(
            cfg,
            "method_kwargs.latent_dim",
            cfg.method_kwargs.state_dim,
        )

        cfg.method_kwargs.sigma_D_dependency = omegaconf_select(
            cfg,
            "method_kwargs.sigma_D_dependency",
            None,
        )
        assert cfg.method_kwargs.sigma_D_dependency in _SIGMA_D_DEPENDENCY, (
            f"Invalid sigma_D_dependency '{cfg.method_kwargs.sigma_D_dependency}'. "
            f"Expected one of {_SIGMA_D_DEPENDENCY}."
        )

        cfg.method_kwargs.sigma_A_dependency = omegaconf_select(
            cfg,
            "method_kwargs.sigma_A_dependency",
            None,
        )
        assert cfg.method_kwargs.sigma_A_dependency in _SIGMA_A_DEPENDENCY, (
            f"Invalid sigma_A_dependency '{cfg.method_kwargs.sigma_A_dependency}'. "
            f"Expected one of {_SIGMA_A_DEPENDENCY}."
        )

        cfg.method_kwargs.loss_function_type = omegaconf_select(
            cfg,
            "method_kwargs.loss_function_type",
            "regular",
        )

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds the learnable parameters of the encoder to the list of learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        kf = self.encoder
        changed_params = kf.slow_params + kf.fast_params + kf.normal_params + kf.super_fast_params + kf.super_slow_params
        for n, p in kf.named_parameters():
            if n.split('.')[0] not in changed_params:
                print(f"Warning: {n} is not in the list of changed parameters.")
        extra_learnable_params = [
            {'params': [p for n, p in kf.named_parameters() if n.split('.')[0] in kf.super_slow_params], 'lr': self.lr * 0.03},
            {'params': [p for n, p in kf.named_parameters() if n.split('.')[0] in kf.slow_params], 'lr': self.lr},
            {'params': [p for n, p in kf.named_parameters() if n.split('.')[0] in kf.normal_params], 'lr': self.lr * 10.0},
            {'params': [p for n, p in kf.named_parameters() if n.split('.')[0] in kf.fast_params], 'lr': self.lr * 100.0},
            {'params': [p for n, p in kf.named_parameters() if n.split('.')[0] in kf.super_fast_params], 'lr': self.lr * 200.0},
        ]
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
        zs_inf_covariances = latents["inferences_covariances"]
        zs_pred_covariances = latents["prediction_covariances"]

        if self.cfg.method_kwargs.loss_function_type == "regular":
            L = kalmanSSL_loss_func
        elif self.cfg.method_kwargs.loss_function_type == "procrustes":
            L = procusteSSL_loss_func

        ssl_loss = L(
            zs_inf,
            zs_pred,
            zs_inf_covariances,
            zs_pred_covariances,
        )

        loss1 = stopgrad_entropy_term(
            zs_inf,
            zs_pred,
            zs_inf_covariances,
            zs_pred_covariances)
        loss2 = kl_entropy_term(
            zs_inf,
            zs_pred,
            zs_inf_covariances,
            zs_pred_covariances)
        cos_sim = compute_gradient_cos_similarity(loss1, loss2, self.encoder)
        self.log("grad_cos_sim", cos_sim, on_step=True, on_epoch=False)



        loss = ssl_loss + decoder_loss

        metrics = {
            "ssl_loss": ssl_loss,
            "loss": loss,
            "inferences_covariances": zs_inf_covariances.mean(),
            "predictions_covariances": zs_pred_covariances.mean(),
            "estimates_covariances": latents["estimates_covariances"].mean(),
            "b_z": self.encoder.b_z.mean(),
            "A": torch.abs(self.encoder.A).mean(),
        }
        self.log_dict(metrics, on_epoch=None)

        return loss

def compute_gradient_cos_similarity(loss1, loss2, model):
    """
    Compute cosine similarity between gradients of loss1 and loss2 w.r.t. model parameters.
    
    Args:
        loss1 (torch.Tensor): scalar loss tensor
        loss2 (torch.Tensor): scalar loss tensor  
        model (torch.nn.Module): the model whose parameters we differentiate w.r.t.
    
    Returns:
        torch.Tensor: cosine similarity (scalar)
    """
    # Get all parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Compute gradients
    grads1 = torch.autograd.grad(loss1, params, retain_graph=True, create_graph=True, allow_unused=True)
    grads2 = torch.autograd.grad(loss2, params, retain_graph=True, create_graph=True, allow_unused=True)
    
    # Flatten and concatenate all gradients into vectors
    vec1 = torch.cat([g.view(-1) for g in grads1 if g is not None])
    vec2 = torch.cat([g.view(-1) for g in grads2 if g is not None])
    
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=1)
    
    return cos_sim.item()  