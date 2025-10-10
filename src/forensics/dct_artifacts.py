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


def blockiness_metric(image: np.ndarray, block_size: int = 8) -> float:
    """Simple blockiness: average absolute difference across block boundaries vs inside blocks."""
    g = _to_gray(image)
    h, w = g.shape
    # vertical boundaries
    vb = np.arange(block_size, w, block_size)
    hb = np.arange(block_size, h, block_size)
    if len(vb) == 0 or len(hb) == 0:
        return 0.0
    # differences across boundaries
    vdiffs = []
    for x in vb:
        vdiffs.append(np.abs(g[:, x - 1] - g[:, x]).mean())
    hdiffs = []
    for y in hb:
        hdiffs.append(np.abs(g[y - 1, :] - g[y, :]).mean())
    boundary_diff = (np.mean(vdiffs) + np.mean(hdiffs)) / 2.0

    # differences inside blocks (1-pixel shifts)
    inside_v = np.abs(g[:, 1:] - g[:, :-1]).mean()
    inside_h = np.abs(g[1:, :] - g[:-1, :]).mean()
    inside = (inside_v + inside_h) / 2.0
    # blockiness is how much stronger boundaries are vs inside
    return float(max(0.0, boundary_diff - inside))


def dct_quantization_strength(image: np.ndarray, block_size: int = 8) -> float:
    """Heuristic: measure energy concentrated at low-frequency DCT bins in 8x8 blocks.

    Strong JPEG compression typically yields higher LF dominance and sparser HF.
    """
    g = _to_gray(image)
    h, w = g.shape
    h8 = h - (h % block_size)
    w8 = w - (w % block_size)
    g = g[:h8, :w8]

    if cv2 is not None:
        # process blocks
        g8 = (g * 255.0).astype(np.uint8)
        g8 = g8.astype(np.float32) - 128.0
        lf_energy = 0.0
        hf_energy = 0.0
        for y in range(0, h8, block_size):
            for x in range(0, w8, block_size):
                block = g8[y : y + block_size, x : x + block_size]
                dct = cv2.dct(block)
                # define LF region (top-left 3x3 excluding DC?) and HF as rest
                lf = dct[:3, :3]
                lf_energy += float(np.sum(lf**2))
                hf = dct
                hf[:3, :3] = 0
                hf_energy += float(np.sum(hf**2))
        denom = hf_energy + 1e-8
        return float(lf_energy / denom)
    else:
        # FFT fallback: compare low vs high frequency energy of full image
        F = np.fft.fft2(g)
        Fshift = np.fft.fftshift(F)
        mag = np.abs(Fshift)
        cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
        r = min(cy, cx) // 8
        lf = mag[cy - r : cy + r, cx - r : cx + r].sum()
        hf = mag.sum() - lf
        return float(lf / (hf + 1e-8))
