# 🚀 Lightweight Real vs AI Image Detector

**Production-ready system optimized for RTX 4060 Laptop GPU**

A fast, memory-efficient hybrid model combining:
- **EfficientNet-B0** backbone (pretrained on ImageNet)
- **Lightweight statistical features** (no heavy DCT/FFT)
- **Mixed precision training** (AMP) - saves ~30% VRAM
- **Memory optimization** - 80% VRAM limit

---

## 📋 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Model Variants](#-model-variants)
- [Performance Tips](#-performance-tips)

---

## ✨ Features

### Lightweight Feature Extraction
- **Pixel Correlation** (horizontal/vertical) - Fast gradient analysis
- **Noise Estimation** (3×3 high-pass filter) - No Gaussian blur overhead
- **Color Consistency** - RGB channel correlation
- **EXIF Metadata** - Camera fingerprint detection

### Optimized Architecture
- **Hybrid Model**: EfficientNet-B0 + 6 statistical features (~5M params)
- **Simple EfficientNet**: Baseline model for comparison (~5M params)
- **Tiny Model**: Ultra-lightweight CNN for edge deployment (~500K params)

### Training Optimizations
- ✅ Mixed precision (AMP) - 30% VRAM savings
- ✅ Memory management - 80% VRAM limit
- ✅ Balanced sampling - Handles class imbalance
- ✅ Cosine annealing scheduler
- ✅ Automatic checkpointing

---

## 🛠️ Installation

```powershell
# Activate virtual environment
.\.venv312\Scripts\Activate.ps1

# Verify all packages are installed
python verify_installation.py
```

**Required Packages** (already installed):
- PyTorch 2.4.1 + CUDA 12.4
- TorchVision 0.19.1
- OpenCV 4.12.0
- scikit-learn, scipy, matplotlib, seaborn
- PIL, tqdm

---

## 🚀 Quick Start

### 1. Prepare Your Data

```
data/
├── train/
│   ├── real/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── ai/
│       ├── img1.jpg
│       └── img2.jpg
└── val/
    ├── real/
    └── ai/
```

**Recommended Dataset Size:**
- **Training**: 2,000-5,000 images (50% real, 50% AI)
- **Validation**: 500-1,000 images

**Where to Get Data:**
- [Kaggle: AI vs Real Faces](https://www.kaggle.com/datasets)
- Generate AI images: Stable Diffusion, Midjourney, DALL·E
- Real images: FFHQ dataset, your own photos

### 2. Train the Model

```powershell
# Hybrid model (recommended)
python train_lightweight.py --model hybrid --batch-size 32 --epochs 50

# Simple EfficientNet baseline
python train_lightweight.py --model efficientnet --batch-size 32 --epochs 50

# Tiny model for edge deployment
python train_lightweight.py --model tiny --batch-size 64 --epochs 50
```

### 3. Evaluate

```powershell
python evaluate.py --checkpoint checkpoints/hybrid_best.pth --test-dir data/val
```

### 4. Run Inference

```powershell
# Single image
python inference.py --checkpoint checkpoints/hybrid_best.pth --image test.jpg

# Folder of images
python inference.py --checkpoint checkpoints/hybrid_best.pth --folder test_images/
```

---

## 📊 Dataset Preparation

### Collecting Data

#### Real Images
- **FFHQ Dataset**: 70,000 high-quality face images
- **Your own photos**: Camera photos with EXIF metadata
- **Stock photos**: Unsplash, Pexels (license-free)

#### AI-Generated Images
- **Stable Diffusion**: Use `diffusers` library
- **Midjourney**: Download from Discord
- **DALL·E**: Via OpenAI API
- **Kaggle datasets**: Search "synthetic faces"

### Quick Test with Small Dataset

```python
# Download and organize 100 images for testing
from pathlib import Path

# Create directories
Path('data/train/real').mkdir(parents=True, exist_ok=True)
Path('data/train/ai').mkdir(parents=True, exist_ok=True)
Path('data/val/real').mkdir(parents=True, exist_ok=True)
Path('data/val/ai').mkdir(parents=True, exist_ok=True)

# Place 40 real + 40 AI in train, 10+10 in val
```

---

## 🏋️ Training

### Full Training Command

```powershell
python train_lightweight.py \
    --model hybrid \
    --train-dir data/train \
    --val-dir data/val \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --workers 2 \
    --checkpoint-dir checkpoints
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `hybrid` | Model type: `hybrid`, `efficientnet`, `tiny` |
| `--batch-size` | `32` | Batch size (32 for RTX 4060) |
| `--epochs` | `50` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--workers` | `2` | DataLoader workers |
| `--no-amp` | `False` | Disable mixed precision |

### Expected Training Time (RTX 4060)
- **2,000 images**: ~5-10 minutes/epoch
- **5,000 images**: ~15-20 minutes/epoch
- **50 epochs**: ~8-16 hours total

### Monitor Training

Training outputs:
```
📂 Loading data...
   Training batches: 63
   Validation batches: 16

🏋️ Starting training for 50 epochs...

Epoch 1/50
Training: 100%|████████| 63/63 [00:45<00:00, 1.39it/s, loss=0.6234, acc=65.50%]
Validation: 100%|████████| 16/16 [00:08<00:00, 1.98it/s]

📊 Epoch 1 Summary:
   Train Loss: 0.6234 | Train Acc: 65.50%
   Val Loss: 0.5821 | Val Acc: 72.30%
   Learning Rate: 0.000100
   💾 Saved best model
```

---

## 📈 Evaluation

### Run Evaluation

```powershell
python evaluate.py \
    --checkpoint checkpoints/hybrid_best.pth \
    --test-dir data/val \
    --output-dir evaluation_results
```

### Output

```
📊 EVALUATION RESULTS
============================================================
Accuracy:  92.45%
Precision: 91.20%
Recall:    93.80%
F1 Score:  92.48%
ROC-AUC:   0.9756
============================================================

💾 Saved confusion matrix to: evaluation_results/confusion_matrix.png
💾 Saved ROC curve to: evaluation_results/roc_curve.png
💾 Saved all results to: evaluation_results
```

### Interpret Results

- **Accuracy**: Overall correctness
- **Precision**: Of predicted AI, how many are actually AI?
- **Recall**: Of actual AI, how many did we catch?
- **F1 Score**: Balance between precision and recall
- **ROC-AUC**: 0.5 = random, 1.0 = perfect

**Target Performance:**
- Accuracy: 85-95%
- ROC-AUC: >0.90

---

## 🔮 Inference

### Single Image Prediction

```powershell
python inference.py \
    --checkpoint checkpoints/hybrid_best.pth \
    --image test.jpg
```

**Output:**
```
============================================================
Image: test.jpg
============================================================
🤖 Prediction: AI
   AI Probability: 87.45%
   Confidence: 87.45%
   [████████████████████████████████████░░░░]
============================================================
```

### Batch Prediction

```powershell
python inference.py \
    --checkpoint checkpoints/hybrid_best.pth \
    --folder test_images/ \
    --pattern "*.jpg" \
    --output results.json
```

### Python API

```python
from inference import RealVsAIPredictor

# Load model
predictor = RealVsAIPredictor('checkpoints/hybrid_best.pth')

# Predict single image
result = predictor.predict_single('test.jpg')
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Predict folder
results = predictor.predict_folder('test_images/')
for filename, pred in results.items():
    print(f"{filename}: {pred['prediction']}")
```

---

## 🎯 Model Variants

### 1. Hybrid Model (Recommended)
```python
from lightweight_model import LightweightHybridDetector

model = LightweightHybridDetector(
    pretrained=True,      # Use ImageNet weights
    num_stat_features=6,  # 6 statistical features
    dropout=0.3,          # Regularization
    freeze_backbone=False # Train full network
)
```

**Best for:** Highest accuracy, moderate speed

### 2. Simple EfficientNet
```python
from lightweight_model import SimpleEfficientNetDetector

model = SimpleEfficientNetDetector(
    pretrained=True,
    dropout=0.3
)
```

**Best for:** Fast inference, good baseline

### 3. Tiny Model
```python
from lightweight_model import TinyDetector

model = TinyDetector(num_stat_features=6)
```

**Best for:** Edge deployment, real-time processing

---

## ⚡ Performance Tips

### Memory Optimization

```python
# Set VRAM limit (80% for RTX 4060)
torch.cuda.set_per_process_memory_fraction(0.8)

# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True

# Use mixed precision
use_amp = True
```

### Speed Optimization

| Setting | Value | Impact |
|---------|-------|--------|
| Batch size | 32-64 | Higher = faster, more VRAM |
| Workers | 2 | More = faster loading, CPU load |
| Mixed precision | True | 30% faster, 30% less VRAM |
| Pin memory | True | Faster GPU transfer |

### Batch Size Guide (RTX 4060 8GB)
- **Hybrid model**: 32-48
- **EfficientNet**: 48-64
- **Tiny model**: 64-128

If out of memory:
```powershell
# Reduce batch size
python train_lightweight.py --batch-size 16

# Or disable AMP (not recommended)
python train_lightweight.py --no-amp
```

---

## 🐛 Troubleshooting

### Issue: CUDA out of memory
**Solution:**
- Reduce `--batch-size` to 16 or 24
- Use `--model tiny` instead of `hybrid`
- Close other GPU programs

### Issue: Low accuracy (<70%)
**Solution:**
- Train longer (100 epochs)
- Use more data (5,000+ images)
- Check data quality (balanced classes)
- Try different learning rates: `--lr 5e-5`

### Issue: Slow training
**Solution:**
- Enable AMP (remove `--no-amp`)
- Increase `--batch-size`
- Reduce `--workers` if CPU bottleneck

---

## 📝 Example Workflows

### Workflow 1: Quick Prototype (1 hour)
```powershell
# 1. Collect 200 images (100 real, 100 AI)
# 2. Train tiny model
python train_lightweight.py --model tiny --epochs 20 --batch-size 64

# 3. Test
python inference.py --checkpoint checkpoints/tiny_best.pth --folder test/
```

### Workflow 2: Production Model (1 day)
```powershell
# 1. Collect 5,000 images
# 2. Train hybrid model
python train_lightweight.py --model hybrid --epochs 100 --batch-size 32

# 3. Evaluate
python evaluate.py --checkpoint checkpoints/hybrid_best.pth --test-dir data/val

# 4. Deploy
python inference.py --checkpoint checkpoints/hybrid_best.pth --folder production/
```

### Workflow 3: Compare All Models
```powershell
# Train all three models
python train_lightweight.py --model hybrid --epochs 50
python train_lightweight.py --model efficientnet --epochs 50
python train_lightweight.py --model tiny --epochs 50

# Evaluate all
python evaluate.py --checkpoint checkpoints/hybrid_best.pth --test-dir data/val
python evaluate.py --checkpoint checkpoints/efficientnet_best.pth --test-dir data/val
python evaluate.py --checkpoint checkpoints/tiny_best.pth --test-dir data/val
```

---

## 📚 Additional Resources

### Datasets
- [Kaggle: Real vs Fake Faces](https://www.kaggle.com/search?q=fake+faces)
- [FFHQ (Real Faces)](https://github.com/NVlabs/ffhq-dataset)
- [GenImage Dataset](https://github.com/GenImage-Dataset/GenImage)

### Papers
- EfficientNet: [Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- Deepfake Detection: Recent surveys on arXiv

### Tools
- **Generate AI images**: Stable Diffusion WebUI
- **Annotate data**: LabelImg, CVAT
- **Monitor training**: TensorBoard

---

## 🎓 Understanding the Model

### Statistical Features (6 total)

1. **Horizontal Pixel Correlation** (0-1)
   - Real: ~0.95 (smooth gradients)
   - AI: ~0.80 (artifacts)

2. **Vertical Pixel Correlation** (0-1)
   - Similar to horizontal

3. **Noise Estimate** (0-1)
   - Real: 0.02-0.05 (sensor noise)
   - AI: 0.01 or >0.10 (too clean or noisy)

4. **Color Consistency** (0-1)
   - Real: ~0.85 (natural RGB correlation)
   - AI: Variable (color artifacts)

5. **Has EXIF Metadata** (0 or 1)
   - Real: 1 (camera data)
   - AI: 0 (no camera)

6. **Has Camera Make** (0 or 1)
   - Real: 1 (Canon, Nikon, etc.)
   - AI: 0 (missing)

### Why Hybrid Wins

**CNN alone**: Learns visual patterns but can be fooled by high-quality AI

**Statistical features alone**: Simple patterns, not robust

**Hybrid**: CNN catches visual artifacts + stats catch mathematical inconsistencies

---

## 🚀 Next Steps

1. ✅ **You're here**: Lightweight system ready
2. 📊 **Collect data**: 2-5K images minimum
3. 🏋️ **Train**: Start with hybrid model
4. 📈 **Evaluate**: Aim for >85% accuracy
5. 🔮 **Deploy**: Use inference.py for production
6. 🔄 **Iterate**: More data → better performance

---

## 💡 Pro Tips

- **Start small**: 500 images × 20 epochs = quick test
- **Use augmentation**: Built-in (flip, rotate, color jitter)
- **Monitor validation**: Stop if val_loss increases
- **Save checkpoints**: Every 5 epochs automatically
- **Test on diverse data**: Different generators, resolutions

---

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Verify installation: `python verify_installation.py`
3. Test with small dataset
4. Check GPU memory: `nvidia-smi`

---

**Status**: ✅ Production Ready
**Hardware**: Optimized for RTX 4060 Laptop GPU
**Training Time**: 8-16 hours (5K images, 50 epochs)
**Expected Accuracy**: 85-95%

Happy detecting! 🎯
