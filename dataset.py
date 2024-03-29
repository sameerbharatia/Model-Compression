import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORKERS = 2

BATCH_SIZE = 128
LEARNING_RATE = 0.001
ETA_MIN = 1e-7

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Pads and then randomly crops
    transforms.RandomHorizontalFlip(),  # Randomly flips the image horizontally
    transforms.RandomRotation(15),  # Randomly rotates the image
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly jitters color
    transforms.ToTensor(),  # Converts to tensor
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),  # Normalizes the dataset
])

val_transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to tensor
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),  # Normalizes the dataset with the same values as the training set
])


train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)

val_set = CIFAR10(root='./data', train=False, download=True, transform=val_transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')