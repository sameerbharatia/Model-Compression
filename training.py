import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import DEVICE, BATCH_SIZE, WORKERS, ETA_MIN
from dataset import train_set, val_set


def train_for_epoch(model, criterion, optimizer, train_data):
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

def evaluate_model(model, criterion, dataset, device=DEVICE):
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

def validation_accuracy(model, criterion, device=DEVICE):
    val_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    return evaluate_model(model, criterion, val_data, device)[1]

def save_model(model, file_path):
    torch.save(model.state_dict(), f'{file_path}.pt')

def load_model(model, file_path):
    model.load_state_dict(torch.load(f'{file_path}.pt'))

def train_val_curve(train_losses, val_losses, train_accuracies, val_accuracies):
    # Plotting the training and validation loss
    plt.figure(figsize=(12, 5))

    # Plot for loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot for accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model(model, criterion, max_epochs, batch_size, learning_rate, file_path, early_stop=False, early_val_accuracy=None):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    max_val_accuracy = 0

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=WORKERS)
    
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=False, num_workers=WORKERS)

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

        print(f'Epoch {epoch:<3}  Train Loss: {train_loss:<6.4f}  Validation Loss: {val_loss:<6.4f}  Train Accuracy: {train_accuracy:<6.2%}  Validation Accuracy: {val_accuracy:<6.2%}')

        if early_stop:
            if val_accuracy >= early_val_accuracy:
                print(f'EARLY STOPPING: Epoch {epoch:<3}')
                break
    
    return train_losses, val_losses, train_accuracies, val_accuracies