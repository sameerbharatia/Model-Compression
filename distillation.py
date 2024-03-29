import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim

from dataset import DEVICE, WORKERS, ETA_MIN
from dataset import train_set, val_set
from training import evaluate_model, save_model


def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    # Soften the logits
    soft_logits_student = F.log_softmax(student_logits / T, dim=1)
    soft_logits_teacher = F.softmax(teacher_logits / T, dim=1)
    
    # Distillation loss
    KL_loss = F.kl_div(soft_logits_student, soft_logits_teacher, reduction='batchmean') * (T * T)
    
    # Traditional cross-entropy loss
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Combined loss
    return alpha * KL_loss + (1 - alpha) * hard_loss

def teach_for_epoch(student_model, teacher_model, train_data, optimizer, T, alpha):
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
        _, predicted = torch.max(student_outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_data)
    accuracy = correct / total

    return avg_loss, accuracy

def teach_model(student_model, teacher_model, T, alpha, criterion, num_epochs, batch_size, learning_rate, file_path):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    max_val_accuracy = 0

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=WORKERS)
    
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=False, num_workers=WORKERS)

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