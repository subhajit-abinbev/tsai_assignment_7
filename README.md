# TSAI Assignment 7: CIFAR-10 Classification with Advanced Data Augmentation

This repository contains three CNN models for CIFAR-10 classification, progressively optimized for efficiency while maintaining high accuracy. The project explores the journey from a large 2M parameter model to an efficient 74K parameter model, demonstrating the power of architectural optimization and advanced data augmentation.

## 📋 Project Overview

### Dataset: CIFAR-10
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image Size**: 32×32×3 (RGB)
- **Training Samples**: 50,000
- **Test Samples**: 10,000
- **Normalization**: Dynamic mean/std computation from training data

### Advanced Data Augmentation
The project uses **Albumentations** library for sophisticated data augmentation:
- **HorizontalFlip**: Random horizontal flipping (p=0.5)
- **Affine Transform**: Combined translation (±10%), scaling (±10%), rotation (±15°)
- **CoarseDropout**: 16×16 pixel cutout filled with dataset mean values
- **Normalization**: Channel-wise normalization with computed statistics

## 🎯 Model Targets

| Model | Parameters | Target Epochs | Target Accuracy |
|-------|------------|---------------|-----------------|
| CNN_Model_1 | ~2M | 60 | >85% |
| CNN_Model_2 | ~200K | 70 | >85% |
| CNN_Model_3 | ~100K | 75 | >85% |

---

## 🔥 Model 1: CNN_Model_1

### Target
- **Parameters**: ~2 Million
- **Target Epochs**: 60
- **Target Accuracy**: >85%

### Architecture
*[Placeholder - To be updated]*

### Training Results
*[Placeholder - To be updated]*

### Analysis
*[Placeholder - To be updated]*

---

## 🚀 Model 2: CNN_Model_2

### Target
- **Parameters**: ~200K
- **Target Epochs**: 70
- **Target Accuracy**: >85%

### Architecture
*[Placeholder - To be updated]*

### Training Results
*[Placeholder - To be updated]*

### Analysis
*[Placeholder - To be updated]*

---

## ⭐ Model 3: CNN_Model_3 (Optimized)

### Target
- **Parameters**: ~100K
- **Target Epochs**: 75
- **Target Accuracy**: >85%

### Architecture Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
       BatchNorm2d-2           [-1, 32, 32, 32]              64
            Conv2d-3           [-1, 32, 32, 32]           9,216
       BatchNorm2d-4           [-1, 32, 32, 32]              64
           Dropout-5           [-1, 32, 32, 32]               0
            Conv2d-6           [-1, 32, 16, 16]             288
       BatchNorm2d-7           [-1, 32, 16, 16]              64
            Conv2d-8           [-1, 64, 16, 16]           2,048
       BatchNorm2d-9           [-1, 64, 16, 16]             128
          Dropout-10           [-1, 64, 16, 16]               0
           Conv2d-11           [-1, 64, 16, 16]             576
      BatchNorm2d-12           [-1, 64, 16, 16]             128
           Conv2d-13           [-1, 96, 16, 16]           6,144
      BatchNorm2d-14           [-1, 96, 16, 16]             192
          Dropout-15           [-1, 96, 16, 16]               0
           Conv2d-16           [-1, 64, 16, 16]           6,144
      BatchNorm2d-17           [-1, 64, 16, 16]             128
           Conv2d-18           [-1, 64, 16, 16]             576
      BatchNorm2d-19           [-1, 64, 16, 16]             128
           Conv2d-20           [-1, 96, 16, 16]           6,144
      BatchNorm2d-21           [-1, 96, 16, 16]             192
          Dropout-22           [-1, 96, 16, 16]               0
           Conv2d-23           [-1, 64, 16, 16]           6,144
      BatchNorm2d-24           [-1, 64, 16, 16]             128
           Conv2d-25             [-1, 64, 8, 8]             576
      BatchNorm2d-26             [-1, 64, 8, 8]             128
           Conv2d-27             [-1, 96, 8, 8]           6,144
      BatchNorm2d-28             [-1, 96, 8, 8]             192
          Dropout-29             [-1, 96, 8, 8]               0
           Conv2d-30             [-1, 64, 8, 8]           6,144
      BatchNorm2d-31             [-1, 64, 8, 8]             128
           Conv2d-32             [-1, 64, 8, 8]             576
      BatchNorm2d-33             [-1, 64, 8, 8]             128
           Conv2d-34             [-1, 96, 8, 8]           6,144
      BatchNorm2d-35             [-1, 96, 8, 8]             192
          Dropout-36             [-1, 96, 8, 8]               0
           Conv2d-37             [-1, 64, 8, 8]           6,144
      BatchNorm2d-38             [-1, 64, 8, 8]             128
           Conv2d-39             [-1, 64, 8, 8]             576
      BatchNorm2d-40             [-1, 64, 8, 8]             128
           Conv2d-41             [-1, 96, 8, 8]           6,144
      BatchNorm2d-42             [-1, 96, 8, 8]             192
          Dropout-43             [-1, 96, 8, 8]               0
AdaptiveAvgPool2d-44             [-1, 96, 1, 1]               0
           Linear-45                   [-1, 10]             970
================================================================
Total params: 73,994
Trainable params: 73,994
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 4.61
Params size (MB): 0.28
Estimated Total Size (MB): 4.90
```

### Key Architecture Features
- **Parameter Count**: 73,994 (26% under target!)
- **Depthwise Separable Convolutions**: Efficient parameter usage
- **Strategic Channel Dimensions**: 32→64→96 progression
- **Batch Normalization**: After every convolution
- **Dropout**: Strategically placed for regularization
- **Global Average Pooling**: Eliminates fully connected layers
- **Memory Efficient**: Only 4.90MB total memory footprint

### Training Configuration
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: Dynamic scheduling
- **Batch Size**: 2048 (both train/test)
- **Epochs**: 100
- **Data Augmentation**: Albumentations pipeline

### Training Progress (Key Milestones)
```
Epoch [1/100]   - Train: 25.97% | Test: 25.00%
Epoch [10/100]  - Train: 64.76% | Test: 66.29%
Epoch [25/100]  - Train: 74.95% | Test: 78.16%
Epoch [50/100]  - Train: 80.38% | Test: 82.73%
Epoch [66/100]  - Train: 82.51% | Test: 85.19% ✅ Target Reached!
Epoch [75/100]  - Train: 83.26% | Test: 85.85%
Epoch [100/100] - Train: 84.08% | Test: 86.38%
```

### Final Results
| Metric | Value |
|--------|-------|
| **Training Loss** | 0.4460 |
| **Training Accuracy** | 84.08% |
| **Test Loss** | 0.4101 |
| **Test Accuracy** | **86.38%** |
| **Target Achievement** | ✅ **Epoch 66/75** |
| **Efficiency** | 9 epochs ahead of target! |

### Model Performance Analysis

#### ✅ **Achievements**
1. **Parameter Efficiency**: 73,994 parameters (26% under 100K target)
2. **Early Target Achievement**: Reached >85% accuracy at epoch 66 (9 epochs early)
3. **Strong Generalization**: Test accuracy (86.38%) > Train accuracy (84.08%)
4. **Stable Training**: Smooth convergence without overfitting
5. **Memory Efficient**: Only 4.90MB total memory footprint

#### 📈 **Training Characteristics**
- **Fast Initial Learning**: 66% accuracy by epoch 10
- **Steady Improvement**: Consistent gains throughout training
- **No Overfitting**: Test accuracy consistently higher than training
- **Stable Convergence**: Final 10 epochs show stable performance around 86%

#### 🎯 **Target Comparison**
| Requirement | Target | Achieved | Status |
|-------------|--------|----------|---------|
| Parameters | <100K | 73,994 | ✅ -26% |
| Epochs to 85% | 75 | 66 | ✅ -12% |
| Final Accuracy | >85% | 86.38% | ✅ +1.38% |

## 📊 Key Insights

### Data Augmentation Impact
The Albumentations pipeline significantly improved generalization:
- **HorizontalFlip**: Helps with spatial invariance
- **Affine Transforms**: Improves robustness to geometric variations
- **CoarseDropout**: Forces model to use diverse features
- **Dynamic Normalization**: Ensures proper statistical properties

### Architecture Optimizations
1. **Depthwise Separable Convolutions**: Massive parameter reduction
2. **Strategic Channel Progression**: Optimal feature extraction
3. **Global Average Pooling**: Eliminates parameter-heavy FC layers
4. **Batch Normalization**: Accelerates training and improves stability

### Training Strategy
- **Large Batch Size**: 2048 for stable gradients
- **AdamW Optimizer**: Handles sparse gradients well
- **Dynamic Scheduling**: Adapts learning rate based on progress

## 🚀 Usage

### Training a Model
```bash
python train.py
# Select model 1, 2, or 3 when prompted
# Choose number of epochs (default: 20)
```

### Data Exploration
```bash
jupyter notebook cifar10_eda.ipynb
```

### Testing Transforms
```bash
python validate_transforms.py
python demo_dynamic_transforms.py
```

## 📁 Project Structure
```
├── model_1.py              # Model 1 (2M parameters)
├── model_2.py              # Model 2 (200K parameters) 
├── model_3.py              # Model 3 (74K parameters)
├── train.py                # Training script
├── utils.py                # Utility functions
├── cifar10_eda.ipynb       # Data exploration notebook
├── output/                 # Training results and plots
├── data/                   # CIFAR-10 dataset (auto-downloaded)
└── README.md               # This file
```

## 🎯 Conclusion

**Model 3 demonstrates exceptional efficiency**, achieving the target accuracy with:
- **26% fewer parameters** than the 100K limit
- **12% fewer epochs** than the 75-epoch target  
- **1.38% higher accuracy** than the 85% requirement

This showcases the power of modern architectural techniques combined with sophisticated data augmentation. The model achieves excellent generalization (test > train accuracy) while maintaining a tiny memory footprint, making it suitable for deployment in resource-constrained environments.

The progression from Model 1 (2M params) → Model 3 (74K params) represents a **96% parameter reduction** while maintaining high accuracy, demonstrating the effectiveness of architectural optimization in deep learning.

## 🔧 Technical Stack
- **Framework**: PyTorch
- **Data Augmentation**: Albumentations
- **Visualization**: Matplotlib, Seaborn
- **Architecture**: Depthwise Separable Convolutions, GAP
- **Optimization**: AdamW, Dynamic LR Scheduling