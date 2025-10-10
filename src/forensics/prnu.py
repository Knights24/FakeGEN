from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        if cv2 is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.mean(axis=2)
    else:
        gray = image
    gray = gray.astype(np.float32)
    if gray.max() > 1.5:
        gray = gray / 255.0
    return gray


def _denoise_bm3d_like(im: np.ndarray, h: float = 3.0) -> np.ndarray:
    """Simple denoising using bilateral filter as BM3D proxy (no external deps).
    This is a placeholder; quality is sufficient to extract a residual signal.
    """
    if cv2 is not None:
        # bilateral preserves edges; parameters tuned lightly
        den = cv2.bilateralFilter((im * 255).astype(np.uint8), d=5, sigmaColor=25, sigmaSpace=7)
        return den.astype(np.float32) / 255.0
    # fallback: box blur
    k = np.ones((5, 5), dtype=np.float32)
    k /= k.sum()
    from .laplacian_noise import cv2_filter2d

    den = cv2_filter2d(im, k)
    return den


def extract_prnu(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Extract PRNU noise residual from a single image.

    Args:
        image: HxW[xC] np.ndarray in uint8 or float
        strength: scaling factor for residual
    Returns:
        residual: HxW float32 zero-mean residual
    """
    gray = _to_gray_float(image)
    den = _denoise_bm3d_like(gray)
    residual = gray - den
    residual -= residual.mean()
    if strength != 1.0:
        residual *= strength
    return residual.astype(np.float32)


def estimate_fingerprint(images: Iterable[np.ndarray]) -> np.ndarray:
    """Estimate a camera fingerprint by averaging PRNU residuals across images."""
    acc = None
    count = 0
    for img in images:
        r = extract_prnu(img)
        if acc is None:
            acc = r.astype(np.float64)
        else:
            acc += r.astype(np.float64)
        count += 1
    if acc is None or count == 0:
        raise ValueError("No images provided for fingerprint estimation")
    fp = acc / float(count)
    # normalize
    std = fp.std() + 1e-8
    fp = (fp - fp.mean()) / std
    return fp.astype(np.float32)


def correlate_with_fingerprint(image: np.ndarray, fingerprint: np.ndarray) -> float:
    """Compute normalized correlation between image PRNU residual and fingerprint."""
    r = extract_prnu(image)
    h, w = fingerprint.shape
    rh, rw = r.shape
    # center crop or pad to match
    h0 = min(h, rh)
    w0 = min(w, rw)
    r_c = r[(rh - h0) // 2 : (rh - h0) // 2 + h0, (rw - w0) // 2 : (rw - w0) // 2 + w0]
    f_c = fingerprint[(h - h0) // 2 : (h - h0) // 2 + h0, (w - w0) // 2 : (w - w0) // 2 + w0]
    r_c = (r_c - r_c.mean()) / (r_c.std() + 1e-8)
    f_c = (f_c - f_c.mean()) / (f_c.std() + 1e-8)
    corr = float((r_c * f_c).mean())
    return corr
