# Targets:
# Set up the environment and build a proper CNN skeleton with Conv + Pooling + FC.
# Make the model lighter by reducing unnecessary parameters (aim <100k).
# Ensure training runs smoothly, no implementation errors.
# Achieve at least 98.5% test accuracy to establish a baseline.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Optimizer configuration
def get_optimizer():
    return optim.AdamW  # Return the actual optimizer class

def get_optimizer_params():
    return {'lr': 0.01, "weight_decay": 1e-4}

# Data augmentation transform 
def get_train_transform(mean, std):
    """
    Training augmentation as per assignment requirements:
    - HorizontalFlip
    - ShiftScaleRotate  
    - CoarseDropout with exact specifications
    """
    # Convert to numpy if tensor
    if hasattr(mean, 'cpu'):
        mean = mean.cpu().numpy()
    if hasattr(std, 'cpu'):
        std = std.cpu().numpy()
    
    mean = np.array(mean)
    std = np.array(std)
    
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(16, 16),
            hole_width_range=(16, 16),
            fill=tuple((mean * 255).astype(int).tolist()),
            p=0.5
        ),
        
        A.Normalize(mean=mean.tolist(), std=std.tolist()),
        ToTensorV2()
    ])

# Test transform without augmentation
def get_test_transform(mean, std):
    return A.Compose([
        A.Normalize(mean=mean.tolist(), std=std.tolist()),
        ToTensorV2()
    ])

# Scheduler configuration for model
def get_scheduler():
    return optim.lr_scheduler.OneCycleLR  # Return the scheduler class

def get_scheduler_params(steps_per_epoch, epochs=20):   
    return {
        'max_lr': 0.12,
        'steps_per_epoch': steps_per_epoch,
        'epochs': epochs,
        'pct_start': 0.15,           # % of cycle spent increasing LR
        'anneal_strategy': 'cos',   # cosine annealing
        'div_factor': 10.0,         # initial_lr = max_lr/div_factor
        'final_div_factor': 1e3     # minimum lr = max_lr/final_div_factor
    }

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()                      # Input size: 3x32x32 (CIFAR-10)

        # Conv Block 1 (Normal Conv)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.01)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.01)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(48)
        self.dropout3 = nn.Dropout(0.01)

        # Conv Block 2 (Depthwise Separable Conv)
        self.conv4 = nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0, bias=False) # Pointwise
        self.bn4 = nn.BatchNorm2d(48)
        self.dropout4 = nn.Dropout(0.02)
        self.conv5 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=48, bias=False) # Depthwise
        self.bn5 = nn.BatchNorm2d(48)
        self.dropout5 = nn.Dropout(0.02)
        self.conv6 = nn.Conv2d(48, 96, kernel_size=1, stride=2, padding=0, bias=False) # Pointwise with stride
        self.bn6 = nn.BatchNorm2d(96)
        self.dropout6 = nn.Dropout(0.02)

        # Conv Block 3 (Diluted Conv)
        self.conv7 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn7 = nn.BatchNorm2d(96)
        self.dropout7 = nn.Dropout(0.04)
        # self.conv8 = nn.Conv2d(128, 128, kernel_size=1, stride=2, padding=0, bias=False)
        # self.bn8 = nn.BatchNorm2d(128)
        # self.dropout8 = nn.Dropout(0.04)

        # Conv Block 4 (Normal Conv)
        self.conv9 = nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(128)
        self.dropout9 = nn.Dropout(0.05)
        # self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn10 = nn.BatchNorm2d(128)
        # self.dropout10 = nn.Dropout(0.05)

        # GAP + FC
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Conv Block 2
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        # Conv Block 3
        x = F.relu(self.bn7(self.conv7(x)))

        # Conv Block 4
        # x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        # x = F.relu(self.bn10(self.conv10(x)))

        # GAP + FC
        x = self.gap(x)
        x = x.view(-1, 128)  # Flatten the tensor
        x = self.fc(x)

        return x