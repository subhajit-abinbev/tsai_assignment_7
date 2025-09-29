#!/usr/bin/env python3
"""
Test CoarseDropout parameters to find the correct ones
"""

import numpy as np
import albumentations as A

def test_coarse_dropout_params():
    """Test different CoarseDropout parameter combinations"""
    dummy_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    mean_values = [0.4914, 0.4822, 0.4465]
    
    print("Testing CoarseDropout parameter combinations...")
    
    # Test 1: Basic parameters
    try:
        transform1 = A.CoarseDropout(
            max_holes=1,
            max_height=16,
            max_width=16,
            p=1.0
        )
        result = transform1(image=dummy_image)
        print("✅ Test 1: Basic parameters work")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test 2: With fill_value
    try:
        transform2 = A.CoarseDropout(
            max_holes=1,
            max_height=16,
            max_width=16,
            fill_value=mean_values,
            p=1.0
        )
        result = transform2(image=dummy_image)
        print("✅ Test 2: With fill_value works")
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    # Test 3: Try alternative parameter names
    try:
        transform3 = A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(16, 16),
            hole_width_range=(16, 16),
            fill_value=mean_values,
            p=1.0
        )
        result = transform3(image=dummy_image)
        print("✅ Test 3: Alternative parameter names work")
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    # Test 4: Check what parameters are actually supported
    try:
        # Let's see what happens with minimal parameters
        transform4 = A.CoarseDropout(p=1.0)
        result = transform4(image=dummy_image)
        print("✅ Test 4: Minimal parameters work")
        print("   Available parameters might be different than expected")
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")

if __name__ == "__main__":
    test_coarse_dropout_params()
    
    # Also check albumentations version
    try:
        import albumentations
        print(f"\nAlbumentations version: {albumentations.__version__}")
    except:
        print("\nCould not determine albumentations version")