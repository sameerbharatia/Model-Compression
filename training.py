import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from dataset import DEVICE, BATCH_SIZE, WORKERS, ETA_MIN
from dataset import train_set, val_set


def train_for_epoch(model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, train_data: DataLoader) -> (float, float):
    '''
    Train the model for one epoch.

    Args:
        model: The neural network model to train.
        criterion: The loss function.
        optimizer: The optimization algorithm.
        train_data: DataLoader for the training data.

    Returns:
        A tuple of average loss and accuracy for this training epoch.
    '''
    running_loss = 0.0
    correct = 0
    total = 0

    model.train()

    for _, data in enumerate(train_data):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_data)
    accuracy = correct / total

    return avg_loss, accuracy

def evaluate_model(model: nn.Module, criterion: nn.Module, dataset: DataLoader, device: torch.device = DEVICE) -> (float, float):
    '''
    Evaluate the model performance on a given dataset.

    Args:
        model: The model to evaluate.
        criterion: The loss function.
        dataset: DataLoader for the dataset to evaluate on.
        device: The device to perform the evaluation.

    Returns:
        A tuple of average loss and accuracy on the dataset.
    '''
    running_loss = 0
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for data in dataset:
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataset)
    accuracy = correct / total

    return avg_loss, accuracy

def validation_accuracy(model: nn.Module, criterion: nn.Module, device: torch.device = DEVICE) -> float:
    '''
    Compute the validation accuracy of the model.

    Args:
        model: The model to evaluate.
        criterion: The loss function.
        device: The device to perform the evaluation.

    Returns:
        The validation accuracy as a float.
    '''
    val_data = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    return evaluate_model(model, criterion, val_data, device)[1]

def save_model(model: nn.Module, file_path: str) -> None:
    '''
    Save the model state to a file.

    Args:
        model: The model to save.
        file_path: Path to the file where the model state is saved.
    '''
    torch.save(model.state_dict(), f'{file_path}.pt')

def load_model(model: nn.Module, file_path: str) -> None:
    '''
    Load the model state from a file.

    Args:
        model: The model to load the state into.
        file_path: Path to the file from where to load the model state.
    '''
    model.load_state_dict(torch.load(f'{file_path}.pt'))

def train_val_curve(train_losses: list, val_losses: list, train_accuracies: list, val_accuracies: list) -> None:
    '''
    Plot training and validation loss and accuracy curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        train_accuracies: List of training accuracies per epoch.
        val_accuracies: List of validation accuracies per epoch.
    '''
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model(model: nn.Module, criterion: nn.Module, max_epochs: int, batch_size: int, learning_rate: float, file_path: str, early_stop: bool = False, early_val_accuracy: float = None) -> (list, list, list, list):
    '''
    Train and validate the model.

    Args:
        model: The model to train and validate.
        criterion: The loss function.
        max_epochs: Maximum number of epochs.
        batch_size: Batch size for training and validation.
        learning_rate: Learning rate for the optimizer.
        file_path: Path to save the best model state.
        early_stop: Whether to stop early if a certain validation accuracy is reached.
        early_val_accuracy: The validation accuracy threshold for early stopping.

    Returns:
        Lists of training losses, validation losses, training accuracies, and validation accuracies.
    '''
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    max_val_accuracy = 0

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=WORKERS)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=ETA_MIN)

    for epoch in range(1, max_epochs + 1):
        train_loss, train_accuracy = train_for_epoch(model, criterion, optimizer, train_loader)
        scheduler.step()
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = evaluate_model(model, criterion, val_loader)
        if val_accuracy > max_val_accuracy:
            save_model(model, file_path)
            max_val_accuracy = val_accuracy

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch:<3} Train Loss: {train_loss:<6.4f} Validation Loss: {val_loss:<6.4f} Train Accuracy: {train_accuracy:<6.2%} Validation Accuracy: {val_accuracy:<6.2%}')

        if early_stop and early_val_accuracy is not None and val_accuracy >= early_val_accuracy:
            print(f'EARLY STOPPING: Epoch {epoch:<3}')
            break
    
    return train_losses, val_losses, train_accuracies, val_accuracies
