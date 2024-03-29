import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    '''A low-rank approximation of a linear layer using singular value decomposition.

    This class decomposes a given linear layer into two linear layers of lower rank,
    potentially reducing the number of parameters significantly while attempting to
    maintain similar functionality.

    Attributes:
        left_linear (nn.Linear): The left linear layer in the decomposition.
        right_linear (nn.Linear): The right linear layer in the decomposition.
    '''
    def __init__(self, original_layer: nn.Linear, delta: int):
        '''
        Initializes the LowRankLinear layer.

        Args:
            original_layer (nn.Linear): The original linear layer to decompose.
            delta (int): A parameter to adjust the rank of the approximation. Higher values reduce the rank further.
        '''
        super(LowRankLinear, self).__init__()
        assert isinstance(original_layer, nn.Linear)
        
        in_dim, out_dim = original_layer.in_features, original_layer.out_features

        # Calculate the rank for the approximation
        # (Boundary rank for exactly zero savings) - delta
        rank = (in_dim * out_dim) // (in_dim + out_dim) - delta
        
        U, S, V = torch.svd_lowrank(original_layer.weight.data, q=rank)
        
        self.left_linear = nn.Linear(in_dim, rank, bias=False)
        self.right_linear = nn.Linear(rank, out_dim, bias=True)
        
        # Initialize the layers with SVD factors
        self.left_linear.weight.data = torch.mm(torch.diag(S), V.t())
        self.right_linear.weight.data = U

        print(f'# of params before: {original_layer.weight.data.numel()}')
        print(f'# of params after: {self.left_linear.weight.data.numel() + self.right_linear.weight.data.numel()}')

        if original_layer.bias is not None:
            self.right_linear.bias.data = original_layer.bias.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the low-rank linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the low-rank linear layers.
        '''
        x = self.left_linear(x)
        x = self.right_linear(x)
        return x

def factor_model(model: nn.Module, delta: int) -> None:
    '''
    Recursively replaces all linear layers in a given model with their low-rank approximations.

    Args:
        model (nn.Module): The neural network model to modify.
        delta (int): A parameter to adjust the rank of the approximation in each linear layer.
    '''
    for name, module in model.named_children():
        factor_model(module, delta)
        
        if isinstance(module, nn.Linear):
            setattr(model, name, LowRankLinear(module, delta))
        elif isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
            for i, submodule in enumerate(module):
                if isinstance(submodule, nn.Linear):
                    module[i] = LowRankLinear(submodule, delta)
