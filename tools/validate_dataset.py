from __future__ import annotations

import argparse
import os
from collections import Counter

try:
    from torchvision.datasets import ImageFolder
except Exception:
    ImageFolder = None


def main():
    p = argparse.ArgumentParser(description="Validate ImageFolder dataset structure")
    p.add_argument("--root", required=True, help="Root folder with class subfolders")
    args = p.parse_args()

    if ImageFolder is None:
        raise RuntimeError("torchvision is required")
    ds = ImageFolder(args.root)
    cnt = Counter([ds.classes[i] for _, i in ds.samples])
    print(f"Classes: {ds.classes}")
    print("Counts:")
    for c in ds.classes:
        print(f"  {c}: {cnt.get(c, 0)}")
    print(f"Total images: {len(ds)}")


if __name__ == "__main__":
    main()
