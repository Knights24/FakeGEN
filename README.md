# Deepfake Detection System

Advanced multi-stream deepfake detection using deep learning and PyTorch.

## Features

- **Multi-Stream Architecture**: Combines spatial, frequency, noise, and metadata analysis
- **EfficientNet-B3 Backbone**: Pre-trained feature extraction
- **Focal Loss**: Handles class imbalance
- **Mixed Precision Training**: FP16 for faster training
- **Trained Model**: Checkpoint available at `checkpoints/best_model.pth`

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- 16GB RAM
- 50GB disk space for datasets

## Quick Start

### 1. Setup Environment

```bash
python -m venv deepfake_env
deepfake_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python verify_setup.py
```

### 3. Dataset Structure

```
datasets/
└── deepdetect-2025/
    ├── train/
    │   ├── real/
    │   └── fake/
    ├── val/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
```

### 4. Train Model

```bash
python train.py
python train.py --epochs 50 --batch_size 16 --lr 0.0001
python train.py --resume checkpoints/best_model.pth
```

### 5. Run Inference

```bash
python inference.py --checkpoint checkpoints/best_model.pth --image test.jpg
```

## Project Structure

```
camera-vs-ai/
├── configs/config.py
├── src/
│   ├── data/dataset.py
│   ├── features/
│   │   ├── dct_extractor.py
│   │   ├── noise_extractor.py
│   │   ├── error_level_analysis.py
│   │   ├── metadata_analyzer.py
│   │   └── pixel_correlation.py
│   ├── models/multistream_detector.py
│   └── training/trainer.py
├── train.py
├── inference.py
├── verify_setup.py
├── checkpoints/best_model.pth
└── requirements.txt
```

## Configuration

Edit `configs/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Batch size |
| `img_size` | 224 | Input size |
| `epochs` | 50 | Training epochs |
| `learning_rate` | 1e-4 | Learning rate |
| `num_workers` | 6 | DataLoader workers |
| `mixed_precision` | True | FP16 training |

## Model Architecture

1. **Spatial Stream**: EfficientNet-B3 RGB features (1536 dims)
2. **Frequency Stream**: DCT convolutions (256 dims)
3. **Noise Stream**: SRM filters (128 dims)
4. **Metadata Stream**: EXIF features (32 dims)

Feature fusion: `F = α·F_spatial + β·F_frequency + γ·F_noise + δ·F_metadata`

## Training

**Loss**: Focal Loss with α=0.25, γ=2.0

**Optimizer**: AdamW with weight decay

**Scheduler**: Cosine Annealing

**Augmentation**: Flips, rotation, color jitter, JPEG compression

## Performance

| Metric | Target |
|--------|--------|
| Accuracy | >95% |
| AUC-ROC | >0.98 |
| F1 Score | >0.94 |

## Troubleshooting

**CUDA OOM**: Reduce batch size `--batch_size 8`

**Slow Training**: Enable mixed precision, adjust workers

**Import Errors**: `pip install -r requirements.txt`

## References

- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [SRM Filters](https://ieeexplore.ieee.org/document/6197267)
- [Focal Loss](https://arxiv.org/abs/1708.02002)
- [timm Library](https://github.com/huggingface/pytorch-image-models)
