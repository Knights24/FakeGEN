# Real vs AI Image Detection System

A comprehensive binary classifier to detect AI-generated images using statistical analysis, frequency domain features, and deep learning.

## 🎯 Goal

Binary classification: **Real** or **AI-generated**

Detection relies on:
- **Statistical noise patterns** - Gaussian residuals, variance analysis
- **Pixel correlation features** - Horizontal, vertical, diagonal correlations
- **Frequency domain inconsistencies** - DCT coefficients, HFER analysis
- **Metadata** - EXIF data, camera fingerprints
- **AI texture artifacts** - GAN fingerprints, PRNU analysis

## ⚙️ System Architecture

### 1. Feature Extraction (`feature_extraction.py`)

#### A. Noise Residual Analysis
```
N(x,y) = I(x,y) - G_σ(I(x,y))
```
- Extracts noise using Gaussian high-pass filter (σ = 1.5)
- Computes noise variance and standard deviation
- Real images: Higher, natural noise variance
- AI images: Lower, uniform noise patterns

#### B. Pixel Correlation
```
ρ = Σ(I_i - Ī)(I_j - Ī) / √(Σ(I_i - Ī)²Σ(I_j - Ī)²)
```
- Computes correlation in 4 directions (H, V, D_main, D_anti)
- Real images: ρ ≈ 0.9+
- AI images: ρ < 0.8

#### C. DCT Frequency Analysis
```
F(u,v) = (1/4)C(u)C(v)Σ I(x,y)cos[(2x+1)uπ/2N]cos[(2y+1)vπ/2N]
E(u,v) = |F(u,v)|²
HFER = Σ_{u,v>f_t} E(u,v) / Σ E(u,v)
```
- High-Frequency Energy Ratio (HFER) analysis
- AI images show unusual frequency distributions
- Block-wise DCT statistics

#### D. PRNU Fingerprint
```
K = Σ(W_i * I_i) / Σ(I_i²)
where W_i = I_i - F(I_i)
```
- Each camera sensor has unique noise pattern
- AI images lack camera-specific fingerprints

#### E. EXIF Metadata
- Camera make/model, GPS, ISO, aperture
- AI images: Empty EXIF or generator tags

### 2. Model Architectures (`models.py`)

#### Option 1: CNN Classifier
```
Input (224×224×3)
    ↓
Conv2D → ReLU → BatchNorm → MaxPool
    ↓
Residual Blocks (ResNet-style)
    ↓
Global Average Pooling
    ↓
Dense(512) → Dropout → Dense(1)
```
**Loss:** Binary Cross Entropy
```
L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

#### Option 2: Hybrid CNN + Statistical Features
```
CNN Branch → f_cnn (deep features)
Statistical Branch → f_stat (noise, correlation, DCT, PRNU)
Metadata Branch → f_meta (EXIF)

h = [f_cnn, f_stat, f_meta]
ŷ = σ(W^T h + b)
```

#### Option 3: Noise Residual CNN (NR-CNN)
Trains directly on noise residual maps for GAN fingerprint detection.

#### Option 4: EfficientNet Baseline
Pre-trained EfficientNet-B0/B1/B2 with custom classifier head.

### 3. Dataset (`dataset.py`)

**Directory Structure:**
```
data/
    real/
        image1.jpg
        image2.jpg
        ...
    ai/
        image1.jpg
        image2.jpg
        ...
```

**Features:**
- Automatic label assignment (Real=0, AI=1)
- Optional statistical feature extraction
- Optional noise residual extraction
- Data augmentation (rotation, flip, color jitter)
- Balanced batch sampling

## 📊 Key Equations

### Image Model
```
I(x,y) = S(x,y) + N(x,y)
```
- I: observed intensity
- S: true signal
- N: noise component

### Noise Extraction
```
N(x,y) = I(x,y) - G_σ(I(x,y))
```
Gaussian blur removes signal, leaves noise.

### Correlation Threshold
```
ρ ≥ 0.9  →  Real camera photo
ρ < 0.8   →  Likely AI-generated
```

### HFER Indicator
```
HFER_real    ≈ 0.2-0.4
HFER_AI      ≈ 0.4-0.7 (unusually high or flat)
```

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/Knights24/FakeGEN.git
cd FakeGEN

# Create virtual environment (Python 3.10-3.12)
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 📦 Requirements

- **PyTorch** >= 2.0.0 (with CUDA for GPU training)
- **torchvision** >= 0.15.0
- **opencv-python** >= 4.8.0
- **scikit-image** >= 0.21.0
- **numpy**, **scipy**, **scikit-learn**
- **matplotlib**, **seaborn** (visualization)
- **piexif**, **exifread** (metadata)

## 🧪 Usage

### 1. Test Feature Extraction

```python
from feature_extraction import FeatureExtractor
import cv2

# Load image
image = cv2.imread('test_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extract features
extractor = FeatureExtractor()
features = extractor.extract_all_features(image, 'test_image.jpg')

print(f"Noise Variance: {features['noise_variance']:.4f}")
print(f"Pixel Correlation: {features['horizontal']:.4f}")
print(f"HFER: {features['hfer']:.4f}")
print(f"PRNU STD: {features['prnu_std']:.4f}")
```

### 2. Test Models

```python
import torch
from models import get_model

# CNN Classifier
model = get_model('cnn')
x = torch.randn(1, 3, 224, 224)
output = model(x)
prob = torch.sigmoid(output)
print(f"AI Probability: {prob.item():.2%}")

# Hybrid Model
model = get_model('hybrid')
stat_feat = torch.randn(1, 14)
meta_feat = torch.randn(1, 7)
output = model(x, stat_feat, meta_feat)

# Noise CNN
model = get_model('noise_cnn')
noise_map = torch.randn(1, 1, 224, 224)
output = model(noise_map)
```

### 3. Prepare Dataset

```python
from dataset import get_dataloaders

# Create dataloaders
train_loader, val_loader = get_dataloaders(
    train_dir='data/train',
    val_dir='data/val',
    batch_size=32,
    model_type='cnn'  # or 'hybrid', 'noise_cnn'
)

# Check data
for batch in train_loader:
    images = batch['image']
    labels = batch['label']
    print(f"Batch: {images.shape}, Labels: {labels.shape}")
    break
```

## 📈 Performance Metrics

```python
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Use **ROC-AUC** to measure robustness across thresholds.

## 🔬 Advanced Techniques

| Technique | Description |
|-----------|-------------|
| **GAN Fingerprint Analysis** | Train on frequency artifacts from StyleGAN, SDXL |
| **JPEG Compression Traces** | Detect double compression, uniform quantization |
| **CLIP-based Separation** | CLIP embeddings + logistic regression |
| **Color Channel Consistency** | RGB correlation analysis - AI often misaligns gradients |

## 📚 Datasets

| Dataset | Description |
|---------|-------------|
| **Kaggle: AI vs Real Faces** | Real + StyleGAN faces |
| **GenImage Dataset** | Large benchmark (real vs AI) |
| **FFHQ** | Real faces dataset (70k images) |
| **Custom AI Images** | Generate from Stable Diffusion, Midjourney, DALL·E |

## 🏗️ Project Structure

```
FakeGEN/
├── feature_extraction.py   # Statistical feature extractors
├── models.py               # Neural network architectures
├── dataset.py              # Dataset and DataLoader
├── train.py               # Training script (to be created)
├── evaluate.py            # Evaluation script (to be created)
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── data/                 # Dataset (user-provided)
    ├── train/
    │   ├── real/
    │   └── ai/
    └── val/
        ├── real/
        └── ai/
```

## 🎓 Citation

If you use this code, please cite:
```bibtex
@software{fakegen2025,
  title={FakeGEN: Real vs AI Image Detection System},
  author={Knights24},
  year={2025},
  url={https://github.com/Knights24/FakeGEN}
}
```

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 🔗 References

1. Noise residual analysis for deepfake detection
2. PRNU-based camera identification
3. Frequency domain analysis for GAN detection
4. EfficientNet: Rethinking Model Scaling
5. Deep Residual Learning for Image Recognition

---

**Note:** This is a research tool. Always verify AI-generated content through multiple methods.
