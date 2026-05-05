import torch
import torch.nn as nn

def VarianceMLP(input_dim, hidden_dim, output_dim, output_activation=None):
    """
    Create a Multi-Layer Perceptron (MLP) model with variance-based activation function.

    Parameters:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden units in the hidden layer.
        output_dim (int): Number of output features.
        output_activation (callable, optional): Activation function to use for output layer.
            Defaults to None, which means no activation function is applied.
    Returns:
        torch.nn.Sequential: The MLP model.
    """

    class Variance(nn.Module):
        def __init__(self):
            super(Variance, self).__init__()
        def forward(self, x):
            return torch.square(x - x.mean(dim=-1, keepdim=True))
    class ElementwiseLinear(nn.Module):
        def __init__(self, input_size: int) -> None:
            super(ElementwiseLinear, self).__init__()
            self.w = nn.Parameter(torch.ones(input_size), requires_grad=True)
        def forward(self, x: torch.tensor) -> torch.tensor:
            return self.w * x
    class PositiveLinear(nn.Module):
        def __init__(self, input_size: int, output_size: int) -> None:
            super(PositiveLinear, self).__init__()
            self.w = nn.Parameter(torch.ones(output_size,input_size)/input_size, requires_grad=True) 
            self.b = nn.Parameter(torch.zeros(output_size), requires_grad=True)
        def forward(self, x: torch.tensor) -> torch.tensor:
            return torch.einsum("ij,...j->...i", torch.abs(self.w), x) + self.b

    layers = []

    layers.append(ElementwiseLinear(input_dim))
    layers.append(Variance())
    
    layers.append(PositiveLinear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    layers.append(PositiveLinear(hidden_dim, output_dim))

    if output_activation is not None:
        layers.append(output_activation())
    
    return nn.Sequential(*layers)
