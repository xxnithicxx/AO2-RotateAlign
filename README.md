# AO2-FIX: Advanced Rotated Object Detection with Matching Degree Loss and CDN

<div align="center">
  <img src="resources/mmrotate-logo.png" width="450"/>
  <div>&nbsp;</div>
</div>

[![PyPI](https://img.shields.io/pypi/v/mmrotate)](https://pypi.org/project/mmrotate)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmrotate.readthedocs.io/en/latest/)
[![license](https://img.shields.io/github/license/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/blob/main/LICENSE)

## 🚀 Project Overview

This project extends the MMRotate framework with two key innovations for improved rotated object detection:

1. **Matching Degree Loss**: A novel loss function that measures the alignment between spatial and feature representations
2. **Conditional DETR with Denoising (CDN)**: Enhanced decoder architecture with denoising training for better query initialization

## ✨ Key Contributions

### 🎯 Matching Degree Loss

The **Matching Degree Loss** introduces a new way to measure the quality of object detection by considering both spatial alignment (SA) and feature alignment (FA):

```
md = α × sa + (1 - α) × fa - |sa - fa|^γ
```

**Key Features:**
- **Spatial Alignment (SA)**: Measures geometric overlap between predicted and ground truth boxes
- **Feature Alignment (FA)**: Evaluates semantic similarity in feature space  
- **Uncertainty Penalty**: The `|sa - fa|^γ` term penalizes inconsistencies between spatial and feature alignments
- **Configurable Balance**: Parameter `α` controls the balance between spatial and feature alignment
- **Adaptive Weighting**: Parameter `γ` adjusts the penalty strength for misalignment

**Implementation Details:**
- Located in `mmrotate/models/losses/matching_degree_loss.py`
- Supports both loss computation and IoU calculation utilities
- Includes `RBboxOverlaps2DIgnoreRot` for rotation-invariant IoU computation
- Integrated with MMRotate's loss registry system

### 🔧 Conditional DETR with Denoising (CDN)

The **CDN Decoder** enhances the standard DETR architecture with advanced query processing and denoising capabilities:

**Key Improvements:**
- **Denoising Training**: Adds noise to ground truth during training to improve robustness
- **Two-Stage Query Processing**: Supports both single-stage and two-stage detection modes
- **Enhanced Reference Point Handling**: Improved spatial reference point propagation
- **Advanced Attention Mechanisms**: Multi-scale deformable attention with conditional queries

**Architecture Components:**
- **Query Denoising**: Implemented in `mmrotate/models/utils/dn_components.py`
- **Enhanced Transformer**: Extended decoder in `mmrotate/models/utils/rotated_transformer.py`
- **Flexible Head Design**: Modified detection head in `mmrotate/models/dense_heads/rotated_deformable_detr_head.py`

## 🏗️ Architecture Overview

```
Input Image
    ↓
Backbone (ResNet-50)
    ↓
Feature Pyramid Network
    ↓
Multi-Scale Deformable Attention Encoder
    ↓
CDN Decoder with Denoising
    ├── Query Initialization
    ├── Denoising Training
    ├── Multi-Head Attention
    └── Reference Point Refinement
    ↓
Detection Head
    ├── Classification Branch
    ├── Regression Branch
    └── Matching Degree Loss
    ↓
Final Predictions
```

## 📊 Performance Improvements

The combination of Matching Degree Loss and CDN decoder provides:

- **Better Convergence**: Matching degree loss provides more stable training signals
- **Improved Localization**: CDN decoder with denoising reduces false positives
- **Enhanced Feature Learning**: Alignment-aware loss encourages better feature representations
- **Robust Training**: Denoising mechanism improves model generalization

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.6+
- CUDA 10.1+

### Setup Instructions

```bash
# Create conda environment
conda create -n ao2-fix python=3.10
conda activate ao2-fix

# Install PyTorch with CUDA support
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install MMCV
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html

# Install MMDetection
pip install mmdet

# Install build requirements
pip install -r requirements/build.txt

# Install MMRotate in development mode
pip install -v -e .
```

## 🚀 Quick Start

### Training with Matching Degree Loss

```bash
# Train with matching degree loss configuration
python tools/train.py configs/deformable_detr/matching_degree_loss.py

# Train with CDN decoder
python tools/train.py configs/deformable_detr/deformable_detr_cdn_r50_16x2_50e_dota.py

# Train with two-stage CDN
python tools/train.py configs/deformable_detr/deformable_detr_twostage_cdn_r50_16x2_50e_dota.py
```

### Configuration Examples

#### Matching Degree Loss Configuration
```python
# In your config file
loss_matching_degree = dict(
    type='MatchingDegreeLoss',
    alpha=0.5,          # Balance between spatial and feature alignment
    gamma=1.0,          # Uncertainty penalty strength
    eps=1e-6,           # Numerical stability
    loss_weight=1.0     # Loss weight in total objective
)
```

#### CDN Decoder Configuration
```python
# CDN with denoising training
bbox_head = dict(
    type='RotatedDeformableDETRHead',
    # ... other parameters ...
    # Denoising configuration
    dn_number=100,                  # Number of denoising queries
    dn_box_noise_scale=0.4,        # Box noise scale
    dn_label_noise_ratio=0.5,      # Label noise ratio
    dn_angle_noise_range=15,       # Angle noise range (degrees)
    dn_labelbook_size=100,         # Label book size
    # Two-stage configuration
    as_two_stage=False,            # Enable two-stage detection
    two_stage_type="standard",     # Two-stage type
)
```

## 📁 Project Structure

```
AO2-FIX/
├── configs/
│   └── deformable_detr/
│       ├── matching_degree_loss.py              # Matching degree loss config
│       ├── deformable_detr_cdn_r50_16x2_50e_dota.py  # CDN config
│       └── deformable_detr_twostage_cdn_r50_16x2_50e_dota.py  # Two-stage CDN
├── mmrotate/
│   ├── models/
│   │   ├── losses/
│   │   │   ├── matching_degree_loss.py          # Matching degree loss implementation
│   │   │   └── __init__.py                      # Updated loss registry
│   │   ├── dense_heads/
│   │   │   └── rotated_deformable_detr_head.py  # Enhanced DETR head
│   │   └── utils/
│   │       ├── dn_components.py                 # Denoising components
│   │       └── rotated_transformer.py           # CDN transformer
├── scripts/
│   ├── train_md.py                              # Training script for matching degree
│   └── setup.sh                                 # Environment setup script
└── tools/
    └── train.py                                 # Main training script
```

## 🔬 Technical Details

### Matching Degree Loss Mathematics

The matching degree loss is computed as:

1. **Spatial Alignment (SA)**: Typically IoU between predicted and ground truth boxes
2. **Feature Alignment (FA)**: Cosine similarity or other feature-based metrics
3. **Matching Degree**: `md = α × sa + (1 - α) × fa - |sa - fa|^γ`
4. **Loss**: `L_md = -log(clamp(md, min=eps))`

### CDN Decoder Process

1. **Query Initialization**: Generate initial object queries
2. **Denoising Preparation**: Add controlled noise to ground truth
3. **Multi-Head Attention**: Process queries with deformable attention
4. **Reference Point Update**: Refine spatial reference points
5. **Output Generation**: Produce final classifications and regressions

## 📈 Experimental Results

The enhanced framework shows improvements in:
- **mAP**: Increased mean Average Precision on DOTA dataset
- **Convergence Speed**: Faster training convergence with matching degree loss
- **Robustness**: Better performance on challenging rotated objects
- **Generalization**: Improved results across different scales and orientations

## 🤝 Contributing

We welcome contributions to improve the AO2-FIX framework:

1. Fork the repository
2. Create a feature branch
3. Implement your improvements
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is released under the [Apache 2.0 license](LICENSE).

## 🙏 Acknowledgments

This work builds upon the excellent [MMRotate](https://github.com/open-mmlab/mmrotate) framework from OpenMMLab. We thank the MMRotate team for providing a solid foundation for rotated object detection research.

---

**Built with ❤️ for the computer vision research community**
