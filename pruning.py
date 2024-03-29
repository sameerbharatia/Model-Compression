import torch
import torch.nn as nn

from dataset import DEVICE
from training import train_model, save_model


def prune_model(model: nn.Module, prune_ratio: float = 0.2) -> dict:
    '''
    Prunes the given model globally by a specified ratio.

    Args:
        model (nn.Module): The model to prune.
        prune_ratio (float): The ratio of weights to prune globally.

    Returns:
        dict: A dictionary of new masks for each pruned parameter.
    '''
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:  # Focus on weights only, excluding biases
            all_weights.append(param.data.view(-1))
    
    all_weights = torch.cat(all_weights).to(DEVICE)
    all_weights = all_weights[all_weights != 0]  # Only look at non-zero weights
    global_threshold = torch.quantile(torch.abs(all_weights), prune_ratio)

    print(f'Pruning threshold: {global_threshold}')

    new_masks = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask = (torch.abs(param.data) > global_threshold).float().to(DEVICE)
            new_masks[name] = mask
            param.data.mul_(mask)

    return new_masks

def prune_apply_masks(model: nn.Module, masks: dict) -> None:
    '''
    Applies given masks to the model, effectively zeroing out pruned weights.

    Args:
        model (nn.Module): The model to apply masks to.
        masks (dict): The masks to apply.
    '''
    for name, param in model.named_parameters():
        if name in masks:
            mask = masks[name].to(DEVICE)
            param.data.mul_(mask)
            param.register_hook(lambda grad, mask=mask: grad * mask)

def percentage_zero_weights(model: nn.Module) -> float:
    '''
    Calculates the percentage of weights that are zero in the model.

    Args:
        model (nn.Module): The model to analyze.

    Returns:
        float: The percentage of weights that are zero.
    '''
    zero_count, total_count = 0, 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            flattened_weights = param.data.view(-1)
            zero_count += torch.sum(flattened_weights == 0).item()
            total_count += flattened_weights.size(0)
    
    return zero_count / total_count

def prune_iterative_train(model: nn.Module, criterion, max_epochs: int, batch_size: int, learning_rate: float, file_path: str, early_val_accuracy: float, prune_ratio: float = 0.2, iterations: int = 10, initial_train: bool = True) -> None:
    '''
    Iteratively prunes and trains a model.

    Args:
        model (nn.Module): The model to prune and train.
        criterion: The loss function used for training.
        max_epochs (int): The maximum number of epochs to train for each iteration.
        batch_size (int): The batch size for training.
        learning_rate (float): The learning rate for training.
        file_path (str): The base file path for saving the model.
        early_val_accuracy (float): The validation accuracy to achieve for early stopping.
        prune_ratio (float): The ratio of weights to prune at each iteration.
        iterations (int): The number of prune-train iterations to perform.
        initial_train (bool): Whether to train the model before starting the prune-train cycles.
    '''
    original_state_dict = {name: param.clone() for name, param in model.state_dict().items()}

    if initial_train:
        train_model(model, criterion, max_epochs, batch_size, learning_rate, file_path, True, early_val_accuracy)
    
    for iteration in range(1, iterations + 1):
        print(f'Pruning iteration {iteration}')
        masks = prune_model(model, prune_ratio)
        prune_apply_masks(model, masks)
        print(f'{percentage_zero_weights(model):.2%} of weights are zero')
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name:
                    param.data = original_state_dict[name].to(DEVICE) * masks[name]

        train_model(model, criterion, max_epochs, batch_size, learning_rate, file_path, True, early_val_accuracy)
        save_model(model, f'{file_path}-{iteration}')

def save_sparse_model(model: nn.Module, file_path: str) -> None:
    '''
    Saves the model's parameters, converting them to sparse tensors if they contain zeros.

    Args:
        model (nn.Module): The model to save.
        file_path (str): The path to save the model to.
    '''
    sparse_model_dict = {}
    for name, param in model.named_parameters():
        if torch.any(param == 0):
            sparse_tensor = param.to_sparse()
            sparse_model_dict[name] = sparse_tensor
        else:
            sparse_model_dict[name] = param
    torch.save(sparse_model_dict, f'{file_path}.pt')

def load_sparse_model(model: nn.Module, file_path: str) -> None:
    '''
    Loads a model's parameters from a file, converting sparse tensors back to dense format as needed.

    Args:
        model (nn.Module): The model to load parameters into.
        file_path (str): The path to load the model from.
    '''
    sparse_model_dict = torch.load(f'{file_path}.pt')
    for name, param in sparse_model_dict.items():
        if param.is_sparse:
            dense_param = param.to_dense()
            model.state_dict()[name].copy_(dense_param)
        else:
            model.state_dict()[name].copy_(param)
