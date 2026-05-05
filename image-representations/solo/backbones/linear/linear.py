import torch
from torch import nn

__all__ = ["Linear"]

class Linear(nn.Module):
    def __init__(self, in_features: int=0, out_features: int=0, positive_weights: bool=False) -> None:
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if positive_weights:
            self.fc.weight.data.clamp_(0)
        self.in_features = in_features
        self.num_features = out_features
        self.positive_weights = positive_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.positive_weights:
            self.fc.weight.data.clamp_(0)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)