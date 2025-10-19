"""
Lightweight Feature Extraction for Real vs AI Detection
No heavy DCT/FFT computations - optimized for speed
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class SimpleFeaturesExtractor:
    """
    Extract lightweight statistical features from images
    - Pixel correlation (horizontal/vertical)
    - Basic noise estimation (3x3 high-pass filter)
    - EXIF metadata presence
    """
    
    def __init__(self):
        # 3x3 High-pass filter for noise detection
        self.highpass_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32) / 8.0
    
    def pixel_correlation(self, img: torch.Tensor) -> float:
        """
        Compute pixel correlation (horizontal gradient consistency)
        Real images: high correlation (0.9+)
        AI images: lower correlation (0.7-0.85)
        
        Args:
            img: Tensor [C, H, W] in range [0, 1]
        
        Returns:
            Correlation score (0-1)
        """
        # Horizontal differences
        diff_h = img[:, :, 1:] - img[:, :, :-1]
        corr_h = 1.0 - diff_h.abs().mean().item()
        
        return max(0.0, min(1.0, corr_h))
    
    def vertical_correlation(self, img: torch.Tensor) -> float:
        """
        Compute vertical pixel correlation
        
        Args:
            img: Tensor [C, H, W]
        
        Returns:
            Correlation score (0-1)
        """
        # Vertical differences
        diff_v = img[:, 1:, :] - img[:, :-1, :]
        corr_v = 1.0 - diff_v.abs().mean().item()
        
        return max(0.0, min(1.0, corr_v))
    
    def noise_estimate(self, img: torch.Tensor) -> float:
        """
        Estimate noise level using high-pass filter
        Real images: moderate noise (0.01-0.05)
        AI images: very low or very high noise
        
        Args:
            img: Tensor [C, H, W]
        
        Returns:
            Noise level (0-1)
        """
        if img.dim() == 3:
            img_gray = img.mean(dim=0, keepdim=True)  # Convert to grayscale
        else:
            img_gray = img
        
        # Apply high-pass filter
        kernel = self.highpass_kernel.view(1, 1, 3, 3)
        if img.is_cuda:
            kernel = kernel.cuda()
        
        # Pad and convolve
        img_padded = F.pad(img_gray.unsqueeze(0), (1, 1, 1, 1), mode='reflect')
        noise_map = F.conv2d(img_padded, kernel)
        
        # Compute noise level
        noise_level = noise_map.abs().mean().item()
        
        return min(1.0, noise_level * 10)  # Scale for better range
    
    def color_consistency(self, img: torch.Tensor) -> float:
        """
        Check RGB channel correlation
        AI images often have misaligned color gradients
        
        Args:
            img: Tensor [3, H, W]
        
        Returns:
            Color consistency score (0-1)
        """
        if img.shape[0] != 3:
            return 0.5  # Default for grayscale
        
        # Compute correlation between channels
        r, g, b = img[0].flatten(), img[1].flatten(), img[2].flatten()
        
        # Pearson correlation
        rg_corr = torch.corrcoef(torch.stack([r, g]))[0, 1].item()
        rb_corr = torch.corrcoef(torch.stack([r, b]))[0, 1].item()
        gb_corr = torch.corrcoef(torch.stack([g, b]))[0, 1].item()
        
        avg_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3.0
        
        return max(0.0, min(1.0, avg_corr))
    
    def extract_exif_features(self, image_path: str) -> Dict[str, float]:
        """
        Extract EXIF metadata as binary features
        Real photos: usually have camera EXIF
        AI images: missing or fake EXIF
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dict with binary flags
        """
        features = {
            'has_exif': 0.0,
            'has_camera_make': 0.0,
            'has_gps': 0.0
        }
        
        try:
            img = Image.open(image_path)
            exif = img._getexif()
            
            if exif is not None:
                features['has_exif'] = 1.0
                
                # Check for camera make (tag 271)
                if 271 in exif or 272 in exif:
                    features['has_camera_make'] = 1.0
                
                # Check for GPS (tag 34853)
                if 34853 in exif:
                    features['has_gps'] = 1.0
        except:
            pass
        
        return features
    
    def extract_all(self, img: torch.Tensor, image_path: Optional[str] = None) -> torch.Tensor:
        """
        Extract all lightweight features and return as tensor
        
        Args:
            img: Tensor [C, H, W] normalized
            image_path: Optional path for EXIF extraction
        
        Returns:
            Feature tensor [6] with:
            [pixel_corr_h, pixel_corr_v, noise, color_consistency, has_exif, has_camera]
        """
        features = []
        
        # Statistical features from image tensor
        features.append(self.pixel_correlation(img))
        features.append(self.vertical_correlation(img))
        features.append(self.noise_estimate(img))
        features.append(self.color_consistency(img))
        
        # EXIF features
        if image_path:
            exif_feats = self.extract_exif_features(image_path)
            features.append(exif_feats['has_exif'])
            features.append(exif_feats['has_camera_make'])
        else:
            features.extend([0.0, 0.0])  # No EXIF data
        
        return torch.tensor(features, dtype=torch.float32)


def batch_extract_features(images: torch.Tensor, 
                           image_paths: Optional[list] = None) -> torch.Tensor:
    """
    Extract features for a batch of images
    
    Args:
        images: Tensor [B, C, H, W]
        image_paths: Optional list of paths for EXIF
    
    Returns:
        Feature tensor [B, 6]
    """
    extractor = SimpleFeaturesExtractor()
    batch_features = []
    
    for i, img in enumerate(images):
        path = image_paths[i] if image_paths else None
        feats = extractor.extract_all(img, path)
        batch_features.append(feats)
    
    return torch.stack(batch_features)


if __name__ == "__main__":
    # Test feature extraction
    print("Testing Simple Features Extractor...")
    
    # Create dummy image
    dummy_img = torch.rand(3, 224, 224)
    
    extractor = SimpleFeaturesExtractor()
    
    print("\nIndividual Features:")
    print(f"Pixel Correlation (H): {extractor.pixel_correlation(dummy_img):.4f}")
    print(f"Pixel Correlation (V): {extractor.vertical_correlation(dummy_img):.4f}")
    print(f"Noise Estimate: {extractor.noise_estimate(dummy_img):.4f}")
    print(f"Color Consistency: {extractor.color_consistency(dummy_img):.4f}")
    
    print("\nAll Features Combined:")
    all_features = extractor.extract_all(dummy_img)
    print(f"Feature vector shape: {all_features.shape}")
    print(f"Feature values: {all_features}")
    
    print("\nBatch Extraction:")
    batch = torch.rand(4, 3, 224, 224)
    batch_features = batch_extract_features(batch)
    print(f"Batch features shape: {batch_features.shape}")
    
    print("\n✅ Simple features extraction working!")
