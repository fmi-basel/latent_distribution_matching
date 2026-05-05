

from .mlp import MLP
import torch.nn as nn

def mlp(method, *args, **kwargs):
    return MLP(kwargs["in_features"], kwargs["hidden_features"], kwargs["out_features"], 
               kwargs.get("num_layers", 1), kwargs.get("activation", nn.ReLU), 
               kwargs.get("output_activation", nn.Identity), kwargs.get("bias", True))

__all__ = ["mlp"]
