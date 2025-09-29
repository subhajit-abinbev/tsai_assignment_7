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
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=0.1,  # ±10% translation
            scale=(0.9, 1.1),       # ±10% scaling
            rotate=(-15, 15),       # ±15° rotation
            p=0.5
        ),
        A.CoarseDropout(
            num_holes_range=(1, 1),    # Exactly 1 hole (min=1, max=1)
            hole_height_range=(16, 16), # Exactly 16px height
            hole_width_range=(16, 16),  # Exactly 16px width
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
    return optim.lr_scheduler.CosineAnnealingLR  # Return the scheduler class

def get_scheduler_params():
    return {
        'T_max': 100,
        'eta_min': 1e-5
    }

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()                      # Input size: 3x32x32 (CIFAR-10)

        # Conv Block 1 (Normal Conv)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        # Conv Block 2 (Depthwise Separable Conv)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=32, bias=False) # Depthwise
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False) # Pointwise
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0, bias=False) # Pointwise with stride
        self.bn6 = nn.BatchNorm2d(128)

        # Conv Block 3 (Diluted Conv)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn7 = nn.BatchNorm2d(256)

        # Conv Block 4 (Normal Conv)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)

        # GAP + FC
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)  # CIFAR-10 has 10 classes

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
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))

        # GAP + FC
        x = self.gap(x)
        x = x.view(-1, 512)  # Flatten the tensor
        x = self.fc(x)

        return x