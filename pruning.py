import torch

from dataset import DEVICE
from training import train_model, save_model


def prune_model(model, prune_ratio=0.2):
    # First, gather all the weights across the model to determine the global pruning threshold
    all_weights = []
    for name, param in model.named_parameters():
        if "weight" in name:  # Focus on weights only, excluding biases
            all_weights.append(param.data.view(-1))
    all_weights = torch.cat(all_weights).to(DEVICE)
    all_weights = all_weights[all_weights != 0] # Only look at non-zero weights
    global_threshold = torch.quantile(torch.abs(all_weights), prune_ratio)

    print(f'Pruning threshold: {global_threshold}')

    # Initialize a dictionary to hold the updated masks
    new_masks = {}

    for name, param in model.named_parameters():
        if "weight" in name:
            # Create a mask where weights that are above the threshold are set to 1, otherwise 0
            mask = (torch.abs(param.data) > global_threshold).float().to(DEVICE)
            new_masks[name] = mask.float()

            # Apply the mask to zero out the pruned weights
            param.data.mul_(mask)

    # Return the updated masks
    return new_masks

def prune_apply_masks(model, masks):
    for name, param in model.named_parameters():
        if name in masks:
            mask = masks[name].to(DEVICE)
            param.data.mul_(mask)  # Apply mask to keep pruned weights at 0
            param.register_hook(lambda grad, mask=mask: grad * mask)  # Keep gradients zeroed out for pruned weights

def percentage_zero_weights(model):
    zero_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        if "weight" in name:  # Focus on weight parameters
            # Flatten the weight tensor to a 1D tensor for easy counting
            flattened_weights = param.data.view(-1)
            # Count zeros in this tensor
            zero_count += torch.sum(flattened_weights == 0).item()
            # Update total count
            total_count += flattened_weights.size(0)
    
    # Calculate percentage of zero weights
    return zero_count / total_count

def prune_iterative_train(model, criterion, max_epochs, batch_size, learning_rate, file_path, early_val_accuracy, prune_ratio=0.2, iterations=10, initial_train=True):
    original_state_dict = {name: param.clone() for name, param in model.state_dict().items()}

    # Initial train
    if initial_train:
        train_model(model, criterion, max_epochs, batch_size, learning_rate, file_path, True, early_val_accuracy)
    
    for iteration in range(1, iterations + 1):
        print(f'Pruning iteration {iteration}')
        # Prune the model and update masks, new masks include previously zeroed weights
        masks = prune_model(model, prune_ratio)

        # Apply updated masks to zero out pruned weights immediately
        prune_apply_masks(model, masks)

        # Optional: Evaluate model performance here to monitor pruning effects
        print(f'{percentage_zero_weights(model):.2%} of weights are zero')
        
        # Reset weights to original values but only for non-pruned connections
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "weight" in name:
                    param.data = original_state_dict[name].to(DEVICE) * masks[name]

        # Train the model
        train_model(model, criterion, max_epochs, batch_size, learning_rate, file_path, True, early_val_accuracy)

        save_model(model, f'{file_path}-{iteration}')

def save_sparse_model(model, file_path):
    sparse_model_dict = {}
    for name, param in model.named_parameters():
        # Convert to sparse tensor if there are zeros
        if torch.any(param == 0):
            sparse_tensor = param.to_sparse()
            sparse_model_dict[name] = sparse_tensor
        else:
            sparse_model_dict[name] = param
    torch.save(sparse_model_dict, f'{file_path}.pt')

def load_sparse_model(model, file_path):
    sparse_model_dict = torch.load(f'{file_path}.pt')
    for name, param in sparse_model_dict.items():
        if param.is_sparse:
            dense_param = param.to_dense()
            model.state_dict()[name].copy_(dense_param)
        else:
            model.state_dict()[name].copy_(param)