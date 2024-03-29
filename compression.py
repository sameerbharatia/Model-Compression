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


def compress_model(distill, prune, factor, quantize):
    model = ConvNet() if distill else DenseNet()
    model = model.to(DEVICE)
    model_file = 'distilled' if distill else 'original'
    
    if prune:
        model_file = 'pruned'
        if distill:
            model_file = 'distilled_pruned'
    
    load_model(model, f'{SAVED_MODELS}/{model_file}')

    if factor:
        factor_model(model, 1)

    if quantize:
        model = quantize_model(model)
    
    return model

def get_leaf_layers(module, layers=None):
    """
    Recursively find all leaf layers in the given module.

    Parameters:
    - module: The root module to search from.
    - layers: (Internal use) List to accumulate leaf layers.

    Returns:
    - List of leaf layers.
    """
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

def get_quantized_model_size(model, pruned):
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
        except:
            for param in layer.parameters():
                size_in_bytes += param.numel() * param.element_size()

    return size_in_bytes

def get_model_size(model, quantized, pruned):
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

combinations = []
for distill in [True, False]:
    for prune in [True, False]:
        for factor in [True, False]:
            for quantize in [True, False]:
                true_conditions = [cond for cond, active in zip(['distill', 'prune', 'factor', 'quantize'],
                                                                [distill, prune, factor, quantize]) if active]
                combination = ', '.join(true_conditions) if true_conditions else "no compression"
                combinations.append(combination)

compression_data = pd.DataFrame({'combination': combinations, 'size': sizes, 'accuracy': accuracies})
compression_data.to_csv('compression_data.csv', index=False)