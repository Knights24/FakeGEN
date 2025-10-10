from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import torch
import numpy as np

try:
    import cv2  # type: ignore[import-not-found]
    from torchvision import transforms  # type: ignore[import-not-found]
except Exception:
    cv2 = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]

from .models import build_model


def load_checkpoint(path: str, model: torch.nn.Module):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])  # type: ignore[index]
    return ckpt


def preprocess(image: "np.ndarray", img_size: int = 224) -> torch.Tensor:
    if transforms is None:
        raise RuntimeError("torchvision is required for preprocessing")
    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tf(image)


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    img_size: int = 224,
    class_names: Optional[List[str]] = None,
):
    if cv2 is None:
        raise RuntimeError("opencv-python is required for reading images")
    im = cv2.imread(image_path)
    if im is None:
        raise RuntimeError(f"Failed to read image {image_path}")
    x = preprocess(im, img_size)
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs_t = torch.softmax(logits, dim=1)[0]
        probs = probs_t.cpu().numpy().tolist()
        pred = int(probs_t.argmax().item())
    label = class_names[pred] if class_names and 0 <= pred < len(class_names) else str(pred)
    return {"path": image_path, "label": label, "probs": probs}


def main():
    p = argparse.ArgumentParser(description="Run inference with trained image classifier")
    p.add_argument("--ckpt", required=True, help="Checkpoint path")
    p.add_argument("--image", help="Single image path")
    p.add_argument("--folder", help="Folder of images (jpg/png)")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--img-size", type=int, default=224)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Temporary model with 2 classes by default; adapt using class names if available in meta file.
    model = build_model(args.backbone, num_classes=2)
    ckpt = load_checkpoint(args.ckpt, model)
    model = model.to(device).eval()
    # Try to load sidecar meta json for class names
    meta_path = os.path.splitext(args.ckpt)[0] + ".meta.json"
    class_names: Optional[List[str]] = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta.get("class_names"), list):
                class_names = [str(x) for x in meta["class_names"]]
        except Exception:
            pass

    results = []
    if args.image:
        results.append(predict_image(model, args.image, device, args.img_size, class_names))
    if args.folder:
        for name in os.listdir(args.folder):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                results.append(predict_image(model, os.path.join(args.folder, name), device, args.img_size, class_names))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
