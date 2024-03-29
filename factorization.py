import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(self, original_layer, delta):
        super(LowRankLinear, self).__init__()
        assert isinstance(original_layer, nn.Linear)
        
        # Perform SVD on the weight matrix of the original layer
        in_dim, out_dim = original_layer.in_features, original_layer.out_features

        rank = (in_dim * out_dim) // (in_dim + out_dim) - delta
        print(f'Reducing linear layer to rank: {rank}')
        
        # rank = min(original_layer.weight.data.shape) - delta
        U, S, V = torch.svd_lowrank(original_layer.weight.data, q=rank)
        
        # Create two new linear layers with lower rank
        self.left_linear = nn.Linear(original_layer.in_features, rank, bias=False)
        self.right_linear = nn.Linear(rank, original_layer.out_features, bias=True)
        
        # Initialize the new layers with the factors from the SVD
        self.left_linear.weight.data = torch.mm(torch.diag(S), V.t())
        self.right_linear.weight.data = U

        print(f'# of params before: {torch.numel(original_layer.weight.data)}')
        print(f'# of params after: {torch.numel(self.left_linear.weight.data) + torch.numel(self.right_linear.weight.data)}\n')
        
        # Copy the bias from the original layer
        if original_layer.bias is not None:
            self.right_linear.bias.data = original_layer.bias.data

    def forward(self, x):
        x = self.left_linear(x)
        x = self.right_linear(x)
        return x

def factor_model(model, delta):
    for name, module in model.named_children():
        # Recursively apply this function to children of current module
        factor_model(module, delta)
        
        if isinstance(module, nn.Linear):
            # Replace the linear layer with a LowRankLinear layer
            setattr(model, name, LowRankLinear(module, delta))
        elif isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
            # For Sequential or ModuleList containers, apply recursively to each module
            for i, submodule in enumerate(module):
                if isinstance(submodule, nn.Linear):
                    module[i] = LowRankLinear(submodule, delta)