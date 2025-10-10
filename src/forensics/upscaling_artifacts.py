from __future__ import annotations

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def _to_gray(image: np.ndarray) -> np.ndarray:
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


def radial_power_spectrum(image: np.ndarray, bins: int = 30) -> np.ndarray:
    """Compute radial power spectrum profile of an image (grayscale).

    Returns an array of length `bins` with average power per radial band.
    """
    g = _to_gray(image)
    F = np.fft.fft2(g)
    Fshift = np.fft.fftshift(F)
    power = np.abs(Fshift) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    rnorm = R / R.max()
    # bin edges
    edges = np.linspace(0, 1.0, bins + 1)
    profile = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        mask = (rnorm >= edges[i]) & (rnorm < edges[i + 1])
        if np.any(mask):
            profile[i] = float(power[mask].mean())
        else:
            profile[i] = 0.0
    return profile.astype(np.float32)


def high_freq_ratio(image: np.ndarray, cutoff: float = 0.6) -> float:
    """Ratio of power beyond cutoff radius to total power."""
    g = _to_gray(image)
    F = np.fft.fft2(g)
    P = np.abs(np.fft.fftshift(F)) ** 2
    h, w = P.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    rnorm = R / R.max()
    hf = P[rnorm >= cutoff].sum()
    tot = P.sum() + 1e-8
    return float(hf / tot)


def aliasing_score(image: np.ndarray) -> float:
    """Measure aliasing via energy concentration in a ring near Nyquist.

    Upscaled images often exhibit elevated ring energy.
    """
    g = _to_gray(image)
    F = np.fft.fft2(g)
    P = np.abs(np.fft.fftshift(F)) ** 2
    h, w = P.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    rnorm = R / R.max()
    ring = (rnorm >= 0.75) & (rnorm <= 0.95)
    inner = rnorm < 0.5
    ring_energy = P[ring].mean() if np.any(ring) else 0.0
    inner_energy = P[inner].mean() if np.any(inner) else 1e-8
    return float(ring_energy / (inner_energy + 1e-8))
