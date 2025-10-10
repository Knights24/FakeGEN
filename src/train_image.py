from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
except Exception as e:  # pragma: no cover
    transforms = None
    ImageFolder = None

from .models import build_model, get_backbone_names


def seed_all(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(img_size: int = 224):
    if transforms is None:
        raise RuntimeError("torchvision is required for training")
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


def create_dataloaders(train_dir: str, val_dir: Optional[str], batch_size: int, num_workers: int, img_size: int = 224):
    train_tf, val_tf = build_transforms(img_size)
    if ImageFolder is None:
        raise RuntimeError("torchvision is required for datasets")
    train_ds = ImageFolder(train_dir, transform=train_tf)
    if val_dir:
        val_ds = ImageFolder(val_dir, transform=val_tf)
    else:
        # split train into train/val
        n = len(train_ds)
        val_size = max(1, int(0.1 * n))
        train_size = n - val_size
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])
        # wrap val_ds with a transform
        val_ds.dataset.transform = val_tf  # type: ignore

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds, val_ds


def train_one_epoch(model, loader, device, criterion, optimizer, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def save_checkpoint(path: str, model, optimizer, epoch: int, best_acc: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_acc": best_acc,
    }, path)


def main():
    p = argparse.ArgumentParser(description="Train image classifier on Real vs AI data")
    p.add_argument("--train", required=True, help="Train folder (ImageFolder format: class subfolders)")
    p.add_argument("--val", help="Validation folder (ImageFolder format)")
    p.add_argument("--out", default="models/image_classifier.pth", help="Checkpoint output path")
    p.add_argument("--backbone", default="resnet18", choices=get_backbone_names())
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="Use mixed precision")
    p.add_argument("--freeze", action="store_true", help="Freeze backbone features")
    args = p.parse_args()

    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, train_ds, val_ds = create_dataloaders(args.train, args.val, args.batch, args.workers, args.img_size)
    num_classes = len(getattr(train_ds, "classes", [])) or 2

    model = build_model(args.backbone, num_classes=num_classes)
    # Wrap backbone to output features (already done in builder)
    if args.freeze:
        for p_ in model.backbone.parameters():
            p_.requires_grad = False

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_acc = 0.0
    class_names = getattr(train_ds, "classes", None)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_checkpoint(args.out, model, optimizer, epoch, best_acc)
            # Also store a sidecar json with metadata
            meta = {
                "epoch": epoch,
                "best_acc": best_acc,
                "backbone": args.backbone,
                "img_size": args.img_size,
                "class_names": class_names,
            }
            with open(os.path.splitext(args.out)[0] + ".meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        print(json.dumps({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_acc": best_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }))

    print(f"Training complete. Best val acc={best_acc:.4f}. Checkpoint saved to {args.out}")
    if class_names:
        print(f"Classes: {class_names}")


if __name__ == "__main__":
    main()
