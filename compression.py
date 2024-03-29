import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from dataset import DEVICE, BATCH_SIZE, LEARNING_RATE
from models import DenseNet, ConvNet
from training import train_model, train_val_curve, validation_accuracy, save_model, load_model
from distillation import teach_model
from pruning import prune_iterative_train, save_sparse_model
from factorization import factor_model
from quantization import quantize_model

SAVED_MODELS = 'saved_models'
criterion = nn.CrossEntropyLoss()

# Train network
original = DenseNet().to(DEVICE)

train_val_curve(*train_model(model=original, 
                             criterion=criterion, 
                             max_epochs=100, 
                             batch_size=BATCH_SIZE, 
                             learning_rate=LEARNING_RATE, 
                             file_path=f'{SAVED_MODELS}/original'))

# Train distilled architecture with raw data first to see how it performs
# Got 74% validation accuracy

smaller = DenseNet().to(DEVICE)

train_val_curve(*train_model(smaller, criterion, max_epochs=500, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, file_path=f'{SAVED_MODELS}/smaller'))

# Distill network
# Got 80% validation accuracy
distilled = ConvNet().to(DEVICE)

train_val_curve(*teach_model(student_model=distilled, 
                             teacher_model=original, 
                             T=5, 
                             alpha=0.7, 
                             criterion=F.cross_entropy, 
                             num_epochs=500, 
                             batch_size=BATCH_SIZE, 
                             learning_rate=LEARNING_RATE, 
                             file_path=f'{SAVED_MODELS}/distilled'))

# Load models with best validation accuracy

load_model(original, f'{SAVED_MODELS}/original')
load_model(distilled, f'{SAVED_MODELS}/distilled')

# Prune original network

desired_val_acc = 0.92

prune_iterative_train(model=original, 
                      criterion=criterion, 
                      max_epochs=100, 
                      batch_size=BATCH_SIZE, 
                      learning_rate=LEARNING_RATE, 
                      file_path=f'{SAVED_MODELS}/pruned', 
                      early_val_accuracy=desired_val_acc,
                      initial_train=False)

# Prune distilled network

desired_val_acc = 0.80

prune_iterative_train(model=distilled, 
                      criterion=criterion, 
                      max_epochs=100, 
                      batch_size=BATCH_SIZE, 
                      learning_rate=LEARNING_RATE, 
                      file_path=f'{SAVED_MODELS}/distilled_pruned', 
                      early_val_accuracy=desired_val_acc,
                      iterations=1,
                      initial_train=False)


def compress_model(distill: bool, prune: bool, factor: bool, quantize: bool) -> nn.Module:
    '''
    Compress a model through distillation, pruning, factorization, and/or quantization based on provided flags.

    Args:
        distill (bool): If True, use a distilled version of the model. Otherwise, use the original version.
        prune (bool): If True, prune the model to remove some weights.
        factor (bool): If True, apply factorization to the model to reduce its complexity.
        quantize (bool): If True, quantize the model to reduce the size of the weights.

    Returns:
        nn.Module: The compressed model after applying the specified compression techniques.

    Note:
        - This function assumes that the models are saved with names that reflect their compression states, such as 'distilled', 'original', 'pruned', and 'distilled_pruned'.
    '''
    # Choose the base model based on distillation flag
    model = ConvNet() if distill else DenseNet()
    model = model.to(DEVICE)
    model_file = 'distilled' if distill else 'original'
    
    # Update model file name based on pruning
    if prune:
        model_file = 'pruned'
        if distill:
            model_file = 'distilled_pruned'
    
    # Load the model
    load_model(model, f'{SAVED_MODELS}/{model_file}')

    # Apply factorization if requested
    if factor:
        factor_model(model, 1)

    # Apply quantization if requested
    if quantize:
        model = quantize_model(model)
    
    return model

def get_leaf_layers(module, layers=None):
    '''
    Recursively find all leaf layers in the given module.

    Parameters:
    - module: The root module to search from.
    - layers: (Internal use) List to accumulate leaf layers.

    Returns:
    - List of leaf layers.
    '''
    if layers is None:
        layers = []

    # If the current module has no children, it's a leaf
    if not list(module.children()):
        layers.append(module)
    else:
        # Recursively apply to children
        for child in module.children():
            get_leaf_layers(child, layers)

    return layers

import torch.nn as nn

def get_quantized_model_size(model: nn.Module, pruned: bool) -> int:
    '''
    Calculate the size of a quantized model in bytes. The size calculation includes
    weights, biases, scales, and zero points for quantized layers, and accounts for
    pruning if applicable.

    Args:
        model (nn.Module): The quantized PyTorch model.
        pruned (bool): Indicates whether the model has been pruned.

    Returns:
        int: The size of the model in bytes.
    '''
    size_in_bytes = 0
    layers = get_leaf_layers(model)

    for layer in layers:
        try:
            weight, bias = layer._weight_bias()
            scale = weight.q_per_channel_scales()
            zero_point = weight.q_per_channel_zero_points()

            num_weights = weight.count_nonzero() if pruned else weight.numel()

            size_in_bytes += num_weights * weight.element_size()
            size_in_bytes += bias.numel() * bias.element_size()
            size_in_bytes += scale.numel() * scale.element_size()
            size_in_bytes += zero_point.numel() * zero_point.element_size()
        except AttributeError:
            # Fallback for non-quantized layers or layers that do not have the expected attributes
            for param in layer.parameters():
                size_in_bytes += param.numel() * param.element_size()

    return size_in_bytes

def get_model_size(model: nn.Module, quantized: bool, pruned: bool) -> int:
    '''
    Calculate the size of a model in bytes, supporting both quantized and non-quantized models.
    For quantized models, it includes the size of weights, biases, scales, and zero points.
    For non-quantized models, it calculates the size based on the weights and potentially pruned elements.
    Additionally, the size of model buffers is included in the calculation.

    Args:
        model (nn.Module): The PyTorch model, either quantized or not.
        quantized (bool): Indicates if the model is quantized.
        pruned (bool): Indicates whether the model has been pruned.

    Returns:
        int: The total size of the model in bytes.
    '''
    size_in_bytes = 0

    if quantized:
        size_in_bytes += get_quantized_model_size(model, pruned)
    else:
        for name, param in model.named_parameters():
            num_elements = param.numel()
            if 'weight' in name and pruned:
                num_elements = param.count_nonzero()
            size_in_bytes += num_elements * param.element_size()
    
    for buffer in model.buffers():
        size_in_bytes += buffer.numel() * buffer.element_size()
    
    return int(size_in_bytes)


# Generate all combinations of compression
accuracies = []
sizes = []

for distill in [True, False]:
    for prune in [True, False]:
        for factor in [True, False]:
            for quantize in [True, False]:
                model = compress_model(distill, prune, factor, quantize)
                
                combination = f'distill={distill},prune={prune},factor={factor},quantize={quantize}'
                accuracy = validation_accuracy(model, criterion, 'cpu' if quantize else DEVICE)
                size = get_model_size(model, quantize, prune)

                accuracies.append(accuracy)
                sizes.append(size)

                print(f'{combination}: Accuracy: {accuracy:.2%} Size: {size / 1e6}MB')

# Save compression data for all combinations
combinations = []
for distill in [True, False]:
    for prune in [True, False]:
        for factor in [True, False]:
            for quantize in [True, False]:
                true_conditions = [cond for cond, active in zip(['distill', 'prune', 'factor', 'quantize'],
                                                                [distill, prune, factor, quantize]) if active]
                combination = ', '.join(true_conditions) if true_conditions else 'no compression'
                combinations.append(combination)

compression_data = pd.DataFrame({'combination': combinations, 'size': sizes, 'accuracy': accuracies})
compression_data.to_csv('compression_data.csv', index=False)