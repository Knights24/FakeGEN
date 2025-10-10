from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert HxWxC or HxW image to grayscale float32 in [0,1]."""
    if image.ndim == 3 and image.shape[2] == 3:
        if cv2 is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.dtype != np.float32 else cv2.cvtColor(
                (np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
            )
        else:
            # fallback: average channels
            gray = image.mean(axis=2)
    else:
        gray = image
    gray = gray.astype(np.float32)
    if gray.max() > 1.5:
        gray = gray / 255.0
    return gray


def variance_of_laplacian(image: np.ndarray) -> float:
    """Compute variance of Laplacian as a sharpness/noise proxy.

    Higher values generally indicate more detail/sharpness; for noise analysis,
    combine with noise maps or compare across regions.
    """
    gray = _to_gray(image)
    if cv2 is None:
        # simple 3x3 Laplacian kernel
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        lap = cv2_filter2d(gray, kernel)
    else:
        lap = cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=3)
    return float(lap.var())


def cv2_filter2d(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Fallback 2D convolution using FFT when cv2 is unavailable."""
    # pad kernel to image size
    s = np.array(src.shape)
    k = np.array(kernel.shape)
    pad = [(0, s[0] - k[0]), (0, s[1] - k[1])]
    ker = np.pad(kernel, pad, mode="constant")
    # FFT-based convolution
    fsrc = np.fft.rfft2(src)
    fker = np.fft.rfft2(ker)
    out = np.fft.irfft2(fsrc * fker)
    # roll to place kernel center
    out = np.roll(out, -k[0] // 2, axis=0)
    out = np.roll(out, -k[1] // 2, axis=1)
    return out.astype(np.float32)


def noise_map_laplacian(image: np.ndarray, window: int = 7) -> np.ndarray:
    """Estimate noise map via local variance of Laplacian response.

    Args:
        image: HxW[xC] array
        window: odd kernel size for local variance computation
    Returns:
        HxW float32 map with higher values where noise/artifacts are higher.
    """
    gray = _to_gray(image)
    if cv2 is None:
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        lap = cv2_filter2d(gray, kernel)
    else:
        lap = cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=3)

    if cv2 is not None:
        mean = cv2.boxFilter(lap, ddepth=-1, ksize=(window, window), normalize=True)
        mean_sq = cv2.boxFilter(lap * lap, ddepth=-1, ksize=(window, window), normalize=True)
        var = np.maximum(mean_sq - mean * mean, 0.0)
    else:
        # naive local mean/variance using uniform kernel via FFT
        k = np.ones((window, window), dtype=np.float32) / (window * window)
        mean = cv2_filter2d(lap, k)
        mean_sq = cv2_filter2d(lap * lap, k)
        var = np.maximum(mean_sq - mean * mean, 0.0)
    return var.astype(np.float32)


def estimate_noise_sigma(image: np.ndarray) -> float:
    """Estimate global noise sigma via median absolute deviation on Laplacian.

    Returns:
        Estimated sigma (0..1 scale when image scaled to 0..1).
    """
    gray = _to_gray(image)
    if cv2 is None:
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        lap = cv2_filter2d(gray, kernel)
    else:
        lap = cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=3)
    med = np.median(lap)
    mad = np.median(np.abs(lap - med))
    # conversion from MAD to sigma for Gaussian: sigma ~= 1.4826 * MAD
    sigma = 1.4826 * mad
    return float(max(0.0, sigma))
