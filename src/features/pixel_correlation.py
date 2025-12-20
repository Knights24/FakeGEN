import numpy as np
import torch
from typing import List, Tuple, Optional


def compute_pixel_correlation(
    image: np.ndarray, 
    displacements: List[Tuple[int, int]] = [(1, 0), (0, 1), (1, 1)]
) -> np.ndarray:
    """
    Calculate spatial correlation coefficient.
    
    Formula:
    ρ = Σ(I(i,j) - Ī)(I(i+Δx, j+Δy) - Ī) / √[Σ(I(i,j) - Ī)² × Σ(I(i+Δx, j+Δy) - Ī)²]
    
    Real images show natural correlation decay; deepfakes exhibit abnormal patterns.
    
    Args:
        image: Input image array [H, W] or [H, W, C]
        displacements: List of (Δx, Δy) tuples for correlation calculation
                      Default: [(1,0), (0,1), (1,1)] for horizontal, vertical, diagonal
        
    Returns:
        correlations: Array of correlation coefficients, one per displacement
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)  # Average RGB channels
    
    image = image.astype(np.float64)
    
    # Calculate mean intensity (Ī)
    mean_intensity = np.mean(image)
    correlations = []
    
    for dx, dy in displacements:
        # Handle edge cases
        if dx == 0 and dy == 0:
            correlations.append(1.0)  # Perfect correlation with self
            continue
        
        # Create shifted versions for correlation computation
        if dx > 0 and dy > 0:
            original = image[:-dx, :-dy]
            shifted = image[dx:, dy:]
        elif dx > 0:
            original = image[:-dx, :]
            shifted = image[dx:, :]
        elif dy > 0:
            original = image[:, :-dy]
            shifted = image[:, dy:]
        else:
            original = image
            shifted = image
        
        # Compute correlation coefficient
        # Numerator: Σ(I(i,j) - Ī)(I(i+Δx, j+Δy) - Ī)
        orig_centered = original - mean_intensity
        shift_centered = shifted - mean_intensity
        numerator = np.sum(orig_centered * shift_centered)
        
        # Denominator: √[Σ(I(i,j) - Ī)² × Σ(I(i+Δx, j+Δy) - Ī)²]
        orig_sq_sum = np.sum(orig_centered ** 2)
        shift_sq_sum = np.sum(shift_centered ** 2)
        denominator = np.sqrt(orig_sq_sum * shift_sq_sum)
        
        # ρ = numerator / denominator (with epsilon for numerical stability)
        correlation = numerator / (denominator + 1e-10)
        correlations.append(float(correlation))
    
    return np.array(correlations)


def compute_pixel_correlation_tensor(
    image_tensor: torch.Tensor, 
    displacements: List[Tuple[int, int]] = [(1, 0), (0, 1), (1, 1)]
) -> torch.Tensor:
    """
    Batch-compatible version for PyTorch tensors.
    
    Args:
        image_tensor: [B, C, H, W] tensor
        displacements: List of (dx, dy) tuples
        
    Returns:
        correlations: [B, num_displacements] tensor
    """
    B, C, H, W = image_tensor.shape
    device = image_tensor.device
    dtype = image_tensor.dtype
    
    # Convert to grayscale using standard weights
    if C == 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=dtype)
        gray = (image_tensor * weights.view(1, 3, 1, 1)).sum(dim=1)  # [B, H, W]
    elif C == 1:
        gray = image_tensor.squeeze(1)  # [B, H, W]
    else:
        gray = image_tensor.mean(dim=1)  # [B, H, W]
    
    correlations_list = []
    
    for dx, dy in displacements:
        if dx == 0 and dy == 0:
            correlations_list.append(torch.ones(B, device=device, dtype=dtype))
            continue
        
        # Create shifted versions
        if dx > 0 and dy > 0:
            original = gray[:, :-dx, :-dy]
            shifted = gray[:, dx:, dy:]
        elif dx > 0:
            original = gray[:, :-dx, :]
            shifted = gray[:, dx:, :]
        elif dy > 0:
            original = gray[:, :, :-dy]
            shifted = gray[:, :, dy:]
        else:
            original = gray
            shifted = gray
        
        # Flatten spatial dimensions
        original_flat = original.reshape(B, -1)  # [B, N]
        shifted_flat = shifted.reshape(B, -1)    # [B, N]
        
        # Compute mean
        mean_orig = original_flat.mean(dim=1, keepdim=True)
        mean_shift = shifted_flat.mean(dim=1, keepdim=True)
        
        # Center the values
        orig_centered = original_flat - mean_orig
        shift_centered = shifted_flat - mean_shift
        
        # Compute correlation
        numerator = (orig_centered * shift_centered).sum(dim=1)
        denom_orig = (orig_centered ** 2).sum(dim=1)
        denom_shift = (shift_centered ** 2).sum(dim=1)
        denominator = torch.sqrt(denom_orig * denom_shift) + 1e-10
        
        correlation = numerator / denominator
        correlations_list.append(correlation)
    
    return torch.stack(correlations_list, dim=1)  # [B, num_displacements]


def compute_correlation_map(
    image: np.ndarray, 
    window_size: int = 16, 
    displacement: Tuple[int, int] = (1, 0)
) -> np.ndarray:
    """
    Compute local correlation map to detect manipulation boundaries.
    
    Manipulated regions often show discontinuities in local correlation.
    
    Args:
        image: Input image [H, W] or [H, W, C]
        window_size: Size of local window for correlation computation
        displacement: Pixel displacement for correlation
        
    Returns:
        correlation_map: [H//window_size, W//window_size] local correlations
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    image = image.astype(np.float64)
    H, W = image.shape
    dx, dy = displacement
    
    # Compute correlation in windows
    h_blocks = H // window_size
    w_blocks = W // window_size
    correlation_map = np.zeros((h_blocks, w_blocks))
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            # Extract window
            y_start = i * window_size
            x_start = j * window_size
            window = image[y_start:y_start + window_size, 
                          x_start:x_start + window_size]
            
            # Compute local correlation
            corr = compute_pixel_correlation(window, [(dx, dy)])
            correlation_map[i, j] = corr[0]
    
    return correlation_map


def extract_correlation_features(image: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive correlation features for deepfake detection.
    
    Args:
        image: Input image [H, W] or [H, W, C]
        
    Returns:
        features: Array of correlation-based features
    """
    # Basic correlations at different displacements
    displacements = [
        (1, 0), (0, 1), (1, 1), (-1, 1),  # 1-pixel
        (2, 0), (0, 2), (2, 2),            # 2-pixel
        (4, 0), (0, 4),                     # 4-pixel
    ]
    
    basic_corr = compute_pixel_correlation(image, displacements)
    
    # Correlation decay (should be natural for real images)
    h_decay = basic_corr[0] - basic_corr[4]  # 1-pixel vs 2-pixel horizontal
    v_decay = basic_corr[1] - basic_corr[5]  # 1-pixel vs 2-pixel vertical
    
    # Additional statistics
    features = np.concatenate([
        basic_corr,
        [h_decay, v_decay],
        [np.mean(basic_corr), np.std(basic_corr)]
    ])
    
    return features
    
    for b in range(B):
        img_np = gray[b].cpu().numpy()
        corr = compute_pixel_correlation(img_np, displacements)
        correlations_list.append(corr)
    
    return torch.from_numpy(np.stack(correlations_list)).float().to(image_tensor.device)
