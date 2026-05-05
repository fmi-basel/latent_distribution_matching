import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers=1, activation=nn.ReLU, output_activation=nn.Identity, bias=True):
        super(MLP, self).__init__()

        self.in_features = input_dim
        self.num_features = output_dim

        layers = []
        if n_hidden_layers == 0:
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            layers.append(activation())
            
            for _ in range(n_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                layers.append(activation())
            
            layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

        layers.append(output_activation())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        return self.model(x)
    
