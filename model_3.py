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

# Optimizer configuration for CNN_Model_3
def get_optimizer():
    return optim.AdamW  # Return the actual optimizer class

def get_optimizer_params():
    return {'lr': 0.01, "weight_decay": 1e-4}

# Data augmentation transform for model_3
def get_train_transform():
    return transforms.Compose([
        transforms.RandomRotation((-10.0, 10.0), fill=(1,)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=(1,)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3, fill=(1,))
    ])

# Scheduler configuration for CNN_Model_3
def get_scheduler():
    return optim.lr_scheduler.StepLR  # Return the scheduler class

def get_scheduler_params():
    return {
        'step_size': 3,
        'gamma': 0.75
    }

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()                      # Input size: 1x28x28

        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 7, 3, padding=1, bias=False)      
        self.bn1 = nn.BatchNorm2d(7)                                
        self.dropout1 = nn.Dropout(0.01)                            

        self.conv2 = nn.Conv2d(7, 11, 3, padding=1, bias=False)    
        self.bn2 = nn.BatchNorm2d(11)                              
        self.dropout2 = nn.Dropout(0.01)                           

        # Transition Block 1
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv Block 2                             
        self.conv3 = nn.Conv2d(11, 11, 3, padding=1, bias=False)    
        self.bn3 = nn.BatchNorm2d(11)                               
        self.dropout3 = nn.Dropout(0.01)

        self.conv4 = nn.Conv2d(11, 15, 3, padding=1, bias=False)    
        self.bn4 = nn.BatchNorm2d(15)                               
        self.dropout4 = nn.Dropout(0.01)                            

        # Transition Block 2
        self.pool2 = nn.MaxPool2d(2, 2)                             

        # Conv Block 3
        self.conv5 = nn.Conv2d(15, 15, 3, padding=1, bias=False)    
        self.bn5 = nn.BatchNorm2d(15)                               
        self.dropout5 = nn.Dropout(0.01)

        self.conv6 = nn.Conv2d(15, 16, 3, padding=1, bias=False)    
        self.bn6 = nn.BatchNorm2d(16)                               
        self.dropout6 = nn.Dropout(0.03)                                                     

        # Output Block
        self.gap = nn.AdaptiveAvgPool2d((1, 1))                     
        self.conv7 = nn.Conv2d(16, 10, 1, padding=0, bias=False)    

    def forward(self, x):
        
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Transition Block 1
        x = self.pool1(x)

        # Conv Block 2
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        # Transition Block 2
        x = self.pool2(x)

        # Conv Block 3
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.dropout6(x)

        # Output Block
        x = self.gap(x) 
        x = self.conv7(x)

        x = x.view(-1, 10*1*1)
        return x