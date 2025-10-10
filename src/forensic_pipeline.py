from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from .forensics import (
        variance_of_laplacian,
        noise_map_laplacian,
        estimate_noise_sigma,
        blockiness_metric,
        dct_quantization_strength,
        radial_power_spectrum,
        high_freq_ratio,
        aliasing_score,
        extract_prnu,
    )
except ImportError:
    # Handle direct script execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.forensics import (
        variance_of_laplacian,
        noise_map_laplacian,
        estimate_noise_sigma,
        blockiness_metric,
        dct_quantization_strength,
        radial_power_spectrum,
        high_freq_ratio,
        aliasing_score,
        extract_prnu,
    )


@dataclass
class ForensicReport:
    path: str
    is_video: bool
    laplacian_var: float
    noise_sigma: float
    blockiness: float
    dct_strength: float
    high_freq_ratio: float
    aliasing_score: float
    prnu_std: float
    radial_profile: Optional[List[float]] = None


def imread(path: str) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    return im


def read_video_frames(path: str, max_frames: int = 32) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    if cv2 is None:
        return frames
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return frames
    count = 0
    while count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        count += 1
    cap.release()
    return frames


def analyze_image(image: np.ndarray, profile_bins: int = 30) -> ForensicReport:
    lap_var = variance_of_laplacian(image)
    sigma = estimate_noise_sigma(image)
    blk = blockiness_metric(image)
    dct_s = dct_quantization_strength(image)
    hf = high_freq_ratio(image)
    alias = aliasing_score(image)
    rp = radial_power_spectrum(image, bins=profile_bins).tolist()
    prnu = extract_prnu(image)
    prnu_std = float(prnu.std())
    return ForensicReport(
        path="",
        is_video=False,
        laplacian_var=lap_var,
        noise_sigma=sigma,
        blockiness=blk,
        dct_strength=dct_s,
        high_freq_ratio=hf,
        aliasing_score=alias,
        prnu_std=prnu_std,
        radial_profile=rp,
    )


def analyze_path(path: str, max_video_frames: int = 16) -> Dict:
    ext = os.path.splitext(path)[1].lower()
    is_video = ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if is_video:
        frames = read_video_frames(path, max_frames=max_video_frames)
        if not frames:
            raise RuntimeError(f"Could not read video frames from {path}")
        # analyze first frame as proxy, and aggregate
        reports = [analyze_image(f) for f in frames]
        # aggregate metrics via median
        agg = {
            "laplacian_var": float(np.median([r.laplacian_var for r in reports])),
            "noise_sigma": float(np.median([r.noise_sigma for r in reports])),
            "blockiness": float(np.median([r.blockiness for r in reports])),
            "dct_strength": float(np.median([r.dct_strength for r in reports])),
            "high_freq_ratio": float(np.median([r.high_freq_ratio for r in reports])),
            "aliasing_score": float(np.median([r.aliasing_score for r in reports])),
            "prnu_std": float(np.median([r.prnu_std for r in reports])),
        }
        return {"path": path, "is_video": True, **agg}
    else:
        im = imread(path)
        if im is None:
            raise RuntimeError(f"Could not read image {path}")
        r = analyze_image(im)
        d = asdict(r)
        d["path"] = path
        return d


def main():
    p = argparse.ArgumentParser(description="Classical forensic feature analysis for image/video")
    p.add_argument("input", help="Path to an image or video file")
    p.add_argument("--out", help="Optional path to save JSON report")
    p.add_argument("--max-frames", type=int, default=16)
    p.add_argument("--bins", type=int, default=30)
    args = p.parse_args()

    result = analyze_path(args.input, max_video_frames=args.max_frames)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
