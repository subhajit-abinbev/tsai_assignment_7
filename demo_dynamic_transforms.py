#!/usr/bin/env python3
"""
Simple demo showing how to use the convenience functions with dynamic mean/std
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import create_cifar10_loaders, compute_mean_std
from model_1 import CNN_Model

def demo_dynamic_transforms():
    """Demo the convenience functions"""
    print("🚀 CIFAR-10 with Dynamic Mean/Std Demo")
    print("=" * 50)
    
    # Method 1: Using convenience function
    print("\n📦 Method 1: Using convenience function")
    trainloader, testloader, mean, std = create_cifar10_loaders(
        batch_size_train=64, 
        batch_size_test=256
    )
    
    print(f"✅ Computed mean: {mean}")
    print(f"✅ Computed std: {std}")
    print(f"✅ Training batches: {len(trainloader)}")
    print(f"✅ Test batches: {len(testloader)}")
    
    # Get a sample batch
    sample_images, sample_labels = next(iter(trainloader))
    print(f"✅ Sample batch shape: {sample_images.shape}")
    print(f"✅ Pixel range: [{sample_images.min():.3f}, {sample_images.max():.3f}]")
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Visualize a few samples
    print("\n🖼️  Visualizing augmented samples...")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(8):
        # Get image and denormalize for visualization
        img = sample_images[i].permute(1, 2, 0)
        img_denorm = img * torch.tensor(std) + torch.tensor(mean)
        img_denorm = torch.clamp(img_denorm, 0, 1)
        
        axes[i].imshow(img_denorm)
        axes[i].set_title(f'{class_names[sample_labels[i]]}', fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('CIFAR-10 with Albumentations (Dynamic Mean/Std)', fontsize=14)
    plt.tight_layout()
    plt.savefig('dynamic_transforms_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test model compatibility
    print("\n🤖 Testing model compatibility...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_Model().to(device)
    
    # Test forward pass
    test_input = sample_images[:4].to(device)
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✅ Model input shape: {test_input.shape}")
    print(f"✅ Model output shape: {output.shape}")
    print(f"✅ Model working correctly!")
    
    # Show the transform info
    print(f"\n📊 Transform Summary:")
    print(f"   • Dataset: CIFAR-10 (50k train, 10k test)")
    print(f"   • Image size: 32x32x3 (RGB)")
    print(f"   • Computed mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"   • Computed std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print(f"   • Augmentations: HorizontalFlip, ShiftScaleRotate, CoarseDropout")
    print(f"   • Normalization: Applied with computed values")
    
    return trainloader, testloader, model

if __name__ == "__main__":
    try:
        print("Starting demo...")
        trainloader, testloader, model = demo_dynamic_transforms()
        print("\n🎉 Demo completed successfully!")
        print("🎉 All components working with dynamic mean/std values!")
        
    except Exception as e:
        print(f"\n💥 Error during demo: {e}")
        import traceback
        traceback.print_exc()