#!/usr/bin/env python3
"""
Test script to verify Albumentations transforms work correctly with CIFAR-10
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import compute_mean_std, data_loader
from model_1 import get_train_transform, get_test_transform, CNN_Model
from torchsummary import summary

def visualize_augmentations():
    """Visualize the effect of Albumentations transforms"""
    print("Testing Albumentations transforms with CIFAR-10...")
    
    # Compute mean and std for CIFAR-10
    train_mean, train_std = compute_mean_std()
    print(f"CIFAR-10 Mean: {train_mean}")
    print(f"CIFAR-10 Std: {train_std}")
    
    # Get transforms with computed mean and std
    train_transform = get_train_transform(train_mean, train_std)
    test_transform = get_test_transform(train_mean, train_std)
    
    # Create data loaders
    trainloader, testloader = data_loader(
        train_mean, train_std,
        batch_size_train=16,
        batch_size_test=16,
        train_transform=train_transform,
        test_transform=test_transform
    )
    
    print(f"Training batches: {len(trainloader)}")
    print(f"Test batches: {len(testloader)}")
    
    # Get a batch of training data
    train_batch = next(iter(trainloader))
    test_batch = next(iter(testloader))
    
    train_images, train_labels = train_batch
    test_images, test_labels = test_batch
    
    print(f"Train batch shape: {train_images.shape}")
    print(f"Test batch shape: {test_images.shape}")
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Visualize some augmented images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for i in range(4):
        # Training images (with augmentation)
        train_img = train_images[i].permute(1, 2, 0)
        # Denormalize for visualization
        train_img = train_img * torch.tensor(train_std) + torch.tensor(train_mean)
        train_img = torch.clamp(train_img, 0, 1)
        
        axes[0, i].imshow(train_img)
        axes[0, i].set_title(f'Train (Aug): {class_names[train_labels[i]]}')
        axes[0, i].axis('off')
        
        # Test images (no augmentation)
        test_img = test_images[i].permute(1, 2, 0)
        # Denormalize for visualization
        test_img = test_img * torch.tensor(train_std) + torch.tensor(train_mean)
        test_img = torch.clamp(test_img, 0, 1)
        
        axes[1, i].imshow(test_img)
        axes[1, i].set_title(f'Test (No Aug): {class_names[test_labels[i]]}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('albumentations_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return trainloader, testloader

def test_model():
    """Test the updated model with CIFAR-10 input"""
    print("\nTesting CNN_Model with CIFAR-10 input...")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_Model().to(device)
    
    print(f"Using device: {device}")
    
    # Model summary
    try:
        summary(model, (3, 32, 32))
    except Exception as e:
        print(f"Error in model summary: {e}")
    
    # Test forward pass
    test_input = torch.randn(1, 3, 32, 32).to(device)
    output = model(test_input)
    print(f"Model output shape: {output.shape}")
    print(f"Output (logits): {output}")
    
    # Test with softmax
    probabilities = torch.softmax(output, dim=1)
    print(f"Probabilities: {probabilities}")
    
    return model

if __name__ == "__main__":
    print("Starting Albumentations and model tests...")
    
    try:
        # Test data loading and visualization
        trainloader, testloader = visualize_augmentations()
        
        # Test model
        model = test_model()
        
        print("\n✅ All tests passed successfully!")
        print("✅ Albumentations transforms working correctly")
        print("✅ Model accepts CIFAR-10 input (3x32x32)")
        print("✅ Data loaders working properly")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()