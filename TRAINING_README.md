# Training: Real vs AI Image Classifier

This adds a ready-to-run pipeline using torchvision backbones (ResNet/EfficientNet).

## 1) Install dependencies (Windows PowerShell)

```powershell
python -m pip install --upgrade pip
# Core
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Project
python -m pip install -r .\forensics_requirements.txt
```

If you don’t have CUDA, omit the `--index-url` or pick the CPU packages from pytorch.org.

## 2) Prepare data

Organize data in ImageFolder format (class subfolders):

```
<train_root>/
  real/
    img1.jpg
    ...
  ai/
    img2.jpg
    ...
```

Optional validation folder in the same structure. If omitted, the script will split train 90/10.

Validate the dataset structure:

```powershell
python -m tools.validate_dataset --root .\data\train
```

## 3) Train

```powershell
python -m src.train_image --train .\data\train --val .\data\val --epochs 10 --batch 32 --backbone resnet18 --out .\models\image_classifier.pth --amp
```

- `--freeze` to freeze the backbone for faster training.
- Checkpoint metadata is saved to `.meta.json` beside the `.pth`.

## 4) Inference

```powershell
python -m src.infer_image --ckpt .\models\image_classifier.pth --image .\some_image.jpg
```

Or a folder of images:

```powershell
python -m src.infer_image --ckpt .\models\image_classifier.pth --folder .\data\val\ai
```

## Tips

- Start with `resnet18` then try `resnet50` for higher capacity.
- If you see dataloader worker issues on Windows, set `--workers 0`.
- For class imbalance, consider oversampling or focal loss (can be added later).
