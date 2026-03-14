# Ceph-SaaS MVP


AI-powered cephalometric analysis platform for orthodontic treatment planning.

## Project Overview
<img src="https://img.sanishtech.com/u/a24fe98a25d5576a35d069e564cacc61.gif" alt="0314-ezgif.com-video-to-gif-converter" loading="lazy" style="max-width:100%;height:auto;">

This project implements an end-to-end machine learning pipeline for automated cephalometric landmark detection using deep learning (U-Net architecture).

## Features

- **Data Ingestion**: Automated loading of Aariz dataset with Senior Orthodontist annotations
- **Deep Learning Model**: U-Net architecture for 29-landmark heatmap regression
- **Training Pipeline**: Optimized for NVIDIA GTX 1650 (4GB VRAM) with gradient clipping
- **Model Verification**: Visualization tools for prediction quality assessment
- **Resume Training**: Power failure recovery with checkpoint resumption

## Project Structure

```
ceph-saas-mvp/
├── backend/
│   ├── core/
│   │   ├── etl/
│   │   │   └── loader.py          # Dataset loading and preprocessing
│   │   └── ml/
│   │       ├── dataset.py         # PyTorch dataset with heatmap generation
│   │       ├── model.py           # U-Net architecture
│   │       └── train.py           # Training script with resume support
│   ├── checkpoints/               # Trained model weights (not in Git)
│   │   ├── best_model.pth        # Best model (epoch 49)
│   │   └── final_model.pth       # Final model (epoch 50)
│   └── tests/
│       ├── visualize_prediction.py  # Inference visualization
│       ├── test_shapes.py           # Integration tests
│       └── debug_nan_loss.py        # Diagnostic tools
├── data/
│   └── Aariz/                     # Dataset (not in Git)
│       ├── train/
│       ├── valid/
│       └── test/
└── README.md
```

## Model Performance

**Training Results:**
- Training Loss: 0.0000
- Validation Loss: 0.0001
- Mean Radial Error: 4.58 pixels (on 512×512 images)

**Best Performing Landmarks:**
- Skeletal landmarks: 1-2px error
- Soft tissue landmarks: 3-5px error
- Dental landmarks: 10-13px error (needs improvement)

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- Git LFS (for model checkpoints)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Walli-Muhammad/ceph-saas.git
cd ceph-saas
```

2. **Create virtual environment**
```bash
python -m venv backend/venv
backend\venv\Scripts\activate  # Windows
# source backend/venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy tqdm
```

4. **Download dataset** (if available)
Place the Aariz dataset in `data/Aariz/`

## Usage

### Training

**Start training from scratch:**
```bash
python backend/core/ml/train.py
```

**Resume training after power failure:**
```bash
python backend/core/ml/train.py --resume
```

### Inference

**Visualize predictions on test image:**
```bash
python backend/tests/visualize_prediction.py
```

**Load model for inference:**
```python
import torch
from backend.core.ml.model import UNet

# Load model
model = UNet(in_channels=1, out_channels=29)
checkpoint = torch.load('backend/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    predictions = model(image_tensor)
```

## Training Configuration

**Hyperparameters:**
- Batch size: 2
- Learning rate: 5e-5
- Epochs: 50
- Optimizer: Adam
- Loss: MSE
- Gradient clipping: max_norm=1.0

**Memory Optimization:**
- Target image size: 512×512
- AMP: Disabled (due to numerical instability)
- GPU memory usage: ~2.4GB

## Model Checkpoints

⚠️ **Note:** Model checkpoint files (`*.pth`) are large (~200MB each) and are managed with Git LFS.

To download the trained models:
```bash
git lfs pull
```

## Development

### Running Tests
```bash
python backend/tests/test_shapes.py
python backend/tests/debug_nan_loss.py
```

### Visualization
```bash
python backend/tests/visualize_prediction.py
```

## Known Issues

1. **AMP Numerical Instability**: Automatic Mixed Precision causes NaN loss. Currently disabled.
2. **Dental Landmark Accuracy**: Tooth cusp landmarks have higher error (10-13px). Needs more training or data augmentation.

## Next Steps

- [ ] Implement inference pipeline for angle calculations (SNA, SNB, ANB)
- [ ] Build FastAPI endpoint for production deployment
- [ ] Add data augmentation for improved robustness
- [ ] Re-enable AMP with newer PyTorch version
- [ ] Create web interface for X-ray upload and analysis

## License

Private project - All rights reserved

## Contact

For questions or collaboration: [Your Contact Info]
