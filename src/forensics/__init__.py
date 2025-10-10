"""
Forensic feature extraction modules.

Includes:
- Laplacian-based noise estimators
- PRNU extraction and correlation utilities
- DCT/JPEG artifact metrics and blockiness
- Upscaling frequency/aliasing measures
- Temporal video model stubs (CNN+LSTM, optical flow)

Each module is import-safe and uses minimal dependencies (numpy, OpenCV, Pillow).
Optional components (torch/torchvision) are used only when available.
"""

from .laplacian_noise import (
    variance_of_laplacian,
    noise_map_laplacian,
    estimate_noise_sigma,
)
from .prnu import (
    extract_prnu,
    estimate_fingerprint,
    correlate_with_fingerprint,
)
from .dct_artifacts import (
    blockiness_metric,
    dct_quantization_strength,
)
from .upscaling_artifacts import (
    radial_power_spectrum,
    high_freq_ratio,
    aliasing_score,
)

__all__ = [
    # Laplacian noise
    "variance_of_laplacian",
    "noise_map_laplacian",
    "estimate_noise_sigma",
    # PRNU
    "extract_prnu",
    "estimate_fingerprint",
    "correlate_with_fingerprint",
    # DCT/JPEG
    "blockiness_metric",
    "dct_quantization_strength",
    # Upscaling
    "radial_power_spectrum",
    "high_freq_ratio",
    "aliasing_score",
]
