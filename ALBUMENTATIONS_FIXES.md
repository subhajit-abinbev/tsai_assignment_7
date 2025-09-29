# Fixed Albumentations Issues

## Issues Identified:
1. ⚠️ **ShiftScaleRotate deprecated**: Warning about using `Affine` instead
2. ⚠️ **CoarseDropout parameter names**: Parameters have changed in newer Albumentations versions

## Fixes Applied:

### 1. Replaced ShiftScaleRotate with Affine
```python
# BEFORE (deprecated)
A.ShiftScaleRotate(
    shift_limit=0.1,
    scale_limit=0.1, 
    rotate_limit=15,
    p=0.5
)

# AFTER (modern)
A.Affine(
    translate_percent=0.1,  # ±10% translation
    scale=(0.9, 1.1),       # ±10% scaling
    rotate=(-15, 15),       # ±15° rotation
    p=0.5
)
```

### 2. Updated CoarseDropout parameters
```python
# BEFORE (old parameter names)
A.CoarseDropout(
    max_holes=1,
    max_height=16,
    max_width=16,
    min_holes=1,
    min_height=16,
    min_width=16,
    fill_value=mean.tolist(),
    mask_fill_value=None,
    p=0.5
)

# AFTER (modern parameter names)
A.CoarseDropout(
    num_holes_range=(1, 1),    # Exactly 1 hole (min=1, max=1)
    hole_height_range=(16, 16), # Exactly 16px height
    hole_width_range=(16, 16),  # Exactly 16px width
    fill_value=mean.tolist(),   # Use computed mean values
    p=0.5
)
```

## Requirements Met:
✅ **HorizontalFlip**: Random horizontal flipping
✅ **ShiftScaleRotate** (via Affine): Combined transformations
✅ **CoarseDropout**: 
  - max_holes = 1 ✅
  - max_height = 16px ✅ 
  - max_width = 16px ✅
  - min_holes = 1 ✅
  - min_height = 16px ✅
  - min_width = 16px ✅
  - fill_value = computed dataset mean ✅
  - mask_fill_value = None ✅

The updated implementation should now work without warnings and provide the exact augmentation behavior you requested.