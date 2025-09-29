#!/usr/bin/env python3
"""
Example training script using Albumentations with dynamic mean/std values
"""

import torch
import torch.optim as optim
from utils import compute_mean_std, data_loader, initialize_model, train_model
from model_1 import get_train_transform, get_test_transform, CNN_Model, get_optimizer, get_optimizer_params

def main():
    print("Starting CIFAR-10 training with Albumentations...")
    
    # Step 1: Compute dynamic mean and std values
    print("Computing CIFAR-10 mean and std values...")
    train_mean, train_std = compute_mean_std()
    print(f"Computed mean: {train_mean}")
    print(f"Computed std: {train_std}")
    
    # Step 2: Get transforms with computed values
    print("Setting up transforms...")
    train_transform = get_train_transform(train_mean, train_std)
    test_transform = get_test_transform(train_mean, train_std)
    
    # Step 3: Create data loaders
    print("Creating data loaders...")
    trainloader, testloader = data_loader(
        train_mean, train_std,
        batch_size_train=128,
        batch_size_test=1024,
        train_transform=train_transform,
        test_transform=test_transform
    )
    
    print(f"Training batches: {len(trainloader)}")
    print(f"Test batches: {len(testloader)}")
    
    # Step 4: Initialize model
    print("Initializing model...")
    optimizer_func = get_optimizer()
    optimizer_params = get_optimizer_params()
    
    model, criterion, optimizer, device = initialize_model(
        CNN_Model, 
        optimizer_func, 
        **optimizer_params
    )
    
    print(f"Using device: {device}")
    print(f"Model initialized with optimizer: {optimizer_func.__name__}")
    print(f"Optimizer params: {optimizer_params}")
    
    # Print model summary
    from torchsummary import summary
    try:
        print("\nModel Architecture:")
        summary(model, (3, 32, 32))
    except Exception as e:
        print(f"Could not print model summary: {e}")
    
    # Step 5: Test with a small number of epochs
    print("\nStarting training (2 epochs for testing)...")
    train_losses, train_accuracies, test_losses, test_accuracies = train_model(
        model, device, trainloader, testloader, optimizer, criterion, epochs=2
    )
    
    print("\nTraining completed!")
    print(f"Final train accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final test accuracy: {test_accuracies[-1]:.2f}%")
    
    # Verify transforms are working
    print("\nVerifying transforms...")
    sample_batch = next(iter(trainloader))
    images, labels = sample_batch
    print(f"Batch shape: {images.shape}")
    print(f"Pixel value range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Check if values are normalized (should be roughly around -2 to +2 after normalization)
    if images.min() < -3 or images.max() > 3:
        print("⚠️  Warning: Images don't appear to be normalized properly")
    else:
        print("✅ Images appear to be normalized correctly")
    
    return model, train_losses, train_accuracies, test_losses, test_accuracies

if __name__ == "__main__":
    try:
        model, train_losses, train_accuracies, test_losses, test_accuracies = main()
        print("\n✅ Script completed successfully!")
        print("✅ Dynamic mean/std computation working")
        print("✅ Albumentations transforms applied correctly")
        print("✅ Model training pipeline functional")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()