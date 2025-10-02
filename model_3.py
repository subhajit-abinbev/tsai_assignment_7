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
    return {'lr': 0.015, 
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999)}

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
    return optim.lr_scheduler.CosineAnnealingLR

def get_scheduler_params(steps_per_epoch, epochs=20):   
    params = {
        'T_max': epochs,
        'eta_min': 9e-5,
    }
    return params

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()                      # Input size: 3x32x32 (CIFAR-10)

        # Conv Block 1 (Initial Conv) (3x32x32 → 32x32x32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.01)

        # Conv Block 2 (Depthwise Separable Conv) (32x32x32 → 16x16x64)    
        self.dwconv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32, bias=False)
        self.dwbn3 = nn.BatchNorm2d(32)
        self.pwconv3 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(0.02)
        
        self.dwconv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2, groups=64, bias=False)
        self.dwbn4 = nn.BatchNorm2d(64)
        self.pwconv4 = nn.Conv2d(64, 96, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(96)
        self.dropout4 = nn.Dropout(0.03)

        # Conv Block 3 (Depthwise Diluted Conv) (16x16x64 → 16x16x128)
        self.conv5 = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.dwconv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=4, dilation=4, groups=64, bias=False)
        self.dwbn6 = nn.BatchNorm2d(64)
        self.pwconv6 = nn.Conv2d(64, 96, kernel_size=1, stride=1, bias=False)
        self.bn6 = nn.BatchNorm2d(96)
        self.dropout6 = nn.Dropout(0.05)

        # Conv Block 4 (DilutedConv) (16x16x128 → 8x8x192)
        self.conv7 = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.dwconv8 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=False)
        self.dwbn8 = nn.BatchNorm2d(64)
        self.pwconv8 = nn.Conv2d(64, 96, kernel_size=1, stride=1, bias=False)
        self.bn8 = nn.BatchNorm2d(96)
        self.dropout8 = nn.Dropout(0.05)

        # (8x8x192 → 8x8x256)
        self.conv9 = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn9 = nn.BatchNorm2d(64)
        self.dwconv10 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.dwbn10 = nn.BatchNorm2d(64)
        self.pwconv10 = nn.Conv2d(64, 96, kernel_size=1, stride=1, bias=False)
        self.bn10 = nn.BatchNorm2d(96)
        self.dropout10 = nn.Dropout(0.06)

        self.conv11 = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn11 = nn.BatchNorm2d(64)
        self.dwconv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.dwbn12 = nn.BatchNorm2d(64)
        self.pwconv12 = nn.Conv2d(64, 96, kernel_size=1, stride=1, bias=False)
        self.bn12 = nn.BatchNorm2d(96)
        self.dropout12 = nn.Dropout(0.07)

        # GAP + FC
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):   
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(x)

        # Conv Block 2
        x = F.relu(self.dwbn3(self.dwconv3(x)))
        x = F.relu(self.bn3(self.pwconv3(x)))
        x = self.dropout3(x)

        # Conv Block 3
        x = F.relu(self.dwbn4(self.dwconv4(x)))
        x = F.relu(self.bn4(self.pwconv4(x)))
        x = self.dropout4(x)

        x = F.relu(self.bn5(self.conv5(x)))

        x = F.relu(self.dwbn6(self.dwconv6(x)))
        x = F.relu(self.bn6(self.pwconv6(x)))
        x = self.dropout6(x)

        x = F.relu(self.bn7(self.conv7(x)))

        # Conv Block 4
        x = F.relu(self.dwbn8(self.dwconv8(x)))
        x = F.relu(self.bn8(self.pwconv8(x)))
        x = self.dropout8(x)

        x = F.relu(self.bn9(self.conv9(x)))

        x = F.relu(self.dwbn10(self.dwconv10(x)))
        x = F.relu(self.bn10(self.pwconv10(x)))
        x = self.dropout10(x)

        x = F.relu(self.bn11(self.conv11(x)))

        x = F.relu(self.dwbn12(self.dwconv12(x)))
        x = F.relu(self.bn12(self.pwconv12(x)))
        x = self.dropout12(x)

        # GAP + FC
        x = self.gap(x)
        x = x.view(-1, 96)
        x = self.fc(x)

        return x