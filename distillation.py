import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import DEVICE, WORKERS, ETA_MIN
from dataset import train_set, val_set
from training import evaluate_model, save_model


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, T: float, alpha: float) -> torch.Tensor:
    '''
    Calculate the knowledge distillation loss between the student and teacher models' logits.

    Args:
        student_logits (torch.Tensor): Logits from the student model.
        teacher_logits (torch.Tensor): Logits from the teacher model.
        labels (torch.Tensor): Ground truth labels.
        T (float): Temperature for softening logits.
        alpha (float): Weight for distillation loss vs. hard loss.

    Returns:
        torch.Tensor: The combined knowledge distillation and cross-entropy loss.
    '''
    soft_logits_student = F.log_softmax(student_logits / T, dim=1)
    soft_logits_teacher = F.softmax(teacher_logits / T, dim=1)
    
    KL_loss = F.kl_div(soft_logits_student, soft_logits_teacher, reduction='batchmean') * (T * T)
    
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * KL_loss + (1 - alpha) * hard_loss

def teach_for_epoch(student_model: torch.nn.Module, teacher_model: torch.nn.Module, train_data: DataLoader, optimizer: optim.Optimizer, T: float, alpha: float) -> tuple[float, float]:
    '''
    Train the student model for one epoch using knowledge distillation from the teacher model.

    Args:
        student_model (torch.nn.Module): The student model to train.
        teacher_model (torch.nn.Module): The teacher model providing guidance.
        train_data (DataLoader): DataLoader providing the training data.
        optimizer (optim.Optimizer): Optimizer used for training the student model.
        T (float): Temperature for softening logits in distillation.
        alpha (float): Weight for distillation loss vs. hard loss.

    Returns:
        tuple[float, float]: Average loss and accuracy over the training data for the epoch.
    '''
    running_loss = 0.0
    correct = 0
    total = 0

    student_model.train()
    teacher_model.eval()
    
    for _, data in enumerate(train_data):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        
        optimizer.zero_grad()
        
        student_outputs = student_model(inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)

        loss = distillation_loss(student_outputs, teacher_outputs, labels, T, alpha)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(student_outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_data)
    accuracy = correct / total

    return avg_loss, accuracy

def teach_model(student_model: torch.nn.Module, teacher_model: torch.nn.Module, T: float, alpha: float, criterion, num_epochs: int, batch_size: int, learning_rate: float, file_path: str) -> tuple[list[float], list[float], list[float], list[float]]:
    '''
    Train the student model using knowledge distillation from the teacher model over multiple epochs.

    Args:
        student_model (torch.nn.Module): The student model to train.
        teacher_model (torch.nn.Module): The teacher model providing guidance.
        T (float): Temperature for softening logits in distillation.
        alpha (float): Weight for distillation loss vs. hard loss.
        criterion: Loss function for evaluation.
        num_epochs (int): Number of epochs to train for.
        batch_size (int): Batch size for training and validation.
        learning_rate (float): Learning rate for the optimizer.
        file_path (str): Path to save the best model.

    Returns:
        tuple[list[float], list[float], list[float], list[float]]: Lists of training losses, validation losses, training accuracies, and validation accuracies.
    '''
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    max_val_accuracy = 0

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=WORKERS)

    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=ETA_MIN)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = teach_for_epoch(student_model, teacher_model, train_loader, optimizer, T, alpha)

        scheduler.step()

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = evaluate_model(student_model, criterion, val_loader)

        if val_accuracy > max_val_accuracy:
            save_model(student_model, file_path)
            max_val_accuracy = val_accuracy

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch:<3}  Train Loss: {train_loss:<6.4f}  Validation Loss: {val_loss:<6.4f}  Train Accuracy: {train_accuracy:<6.2%}  Validation Accuracy: {val_accuracy:<6.2%}')
    
    return train_losses, val_losses, train_accuracies, val_accuracies