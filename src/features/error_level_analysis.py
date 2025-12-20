import io
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import Union, Tuple, Optional


class ELAExtractor:
    """
    Error Level Analysis for compression artifact detection.
    
    ELA reveals areas of an image that have been modified after the initial
    compression. Authentic images show uniform compression levels, while
    manipulated images show inconsistencies in different regions.
    
    Attributes:
        quality: JPEG recompression quality (default=90)
    """
    
    def __init__(self, quality: int = 90):
        """
        Args:
            quality: JPEG quality level for recompression (0-100)
                    Higher values (90-95) detect subtle manipulations
                    Lower values (70-80) detect heavier manipulations
        """
        self.quality = quality
    
    def extract_ela(self, image_path: str) -> np.ndarray:
        """
        Extract ELA map from image file.
        
        Formula: ELA(x,y) = |I_original(x,y) - I_recompressed(x,y)|
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ela_map: Normalized difference map [H, W, 3] in range [0, 255]
                    Higher values indicate potential manipulation
        """
        # Load original image
        original = Image.open(image_path).convert('RGB')
        original_array = np.array(original, dtype=np.float32)
        
        # Recompress at specified quality level
        buffer = io.BytesIO()
        original.save(buffer, 'JPEG', quality=self.quality)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert('RGB')
        recompressed_array = np.array(recompressed, dtype=np.float32)
        
        # Calculate absolute difference: |I_original - I_recompressed|
        ela_map = np.abs(original_array - recompressed_array)
        
        # Normalize to [0, 255] for visualization and analysis
        if ela_map.max() > 0:
            ela_map = (ela_map / ela_map.max() * 255).astype(np.uint8)
        else:
            ela_map = ela_map.astype(np.uint8)
        
        return ela_map
    
    def extract_ela_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """
        Extract ELA map from numpy array.
        
        Args:
            image_array: RGB image array [H, W, 3] in range [0, 255]
            
        Returns:
            ela_map: Normalized ELA map [H, W, 3]
        """
        # Convert to PIL Image
        img_pil = Image.fromarray(image_array.astype(np.uint8))
        
        # Recompress
        buffer = io.BytesIO()
        img_pil.save(buffer, 'JPEG', quality=self.quality)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert('RGB')
        recompressed_array = np.array(recompressed, dtype=np.float32)
        
        # Calculate ELA
        ela_map = np.abs(image_array.astype(np.float32) - recompressed_array)
        
        # Normalize
        if ela_map.max() > 0:
            ela_map = (ela_map / ela_map.max() * 255).astype(np.uint8)
        else:
            ela_map = ela_map.astype(np.uint8)
        
        return ela_map
    
    def extract_ela_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract ELA from tensor (for batch processing in training).
        
        Args:
            image_tensor: [B, C, H, W] tensor in range [0, 1]
            
        Returns:
            ela_tensor: [B, C, H, W] tensor with ELA maps
        """
        B, C, H, W = image_tensor.shape
        device = image_tensor.device
        dtype = image_tensor.dtype
        ela_maps = []
        
        for b in range(B):
            # Convert tensor to numpy (denormalize if needed)
            img_np = image_tensor[b].permute(1, 2, 0).cpu().numpy()
            
            # Scale to 0-255 if in 0-1 range
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            
            # Extract ELA
            ela_map = self.extract_ela_from_array(img_np)
            
            # Convert back to tensor [C, H, W] in range [0, 1]
            ela_tensor = torch.from_numpy(ela_map).permute(2, 0, 1).float() / 255.0
            ela_maps.append(ela_tensor)
        
        return torch.stack(ela_maps).to(device=device, dtype=dtype)
    
    def get_ela_statistics(self, ela_map: np.ndarray) -> dict:
        """
        Compute statistical features from ELA map.
        
        Args:
            ela_map: ELA map [H, W, 3]
            
        Returns:
            stats: Dictionary with mean, std, max, and histogram features
        """
        # Convert to grayscale for analysis
        if len(ela_map.shape) == 3:
            gray = np.mean(ela_map, axis=2)
        else:
            gray = ela_map
        
        stats = {
            'mean': float(np.mean(gray)),
            'std': float(np.std(gray)),
            'max': float(np.max(gray)),
            'min': float(np.min(gray)),
            'median': float(np.median(gray)),
            # Percentage of high ELA pixels (potential manipulation)
            'high_ela_ratio': float(np.mean(gray > 50)),
            # Variance ratio (manipulated regions often have higher variance)
            'variance': float(np.var(gray)),
        }
        
        return stats


class ELAConvExtractor(nn.Module):
    """
    Learnable ELA feature extractor using convolutions.
    
    Combines ELA extraction with convolutional processing to learn
    discriminative features from compression artifacts.
    """
    
    def __init__(self, out_features: int = 64, quality: int = 90):
        super().__init__()
        self.ela_extractor = ELAExtractor(quality=quality)
        
        # Convolutional processing of ELA features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(64, out_features)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract learnable ELA features.
        
        Args:
            images: [B, 3, H, W] input images
            
        Returns:
            features: [B, out_features] ELA-based features
        """
        # Extract ELA maps
        ela_maps = self.ela_extractor.extract_ela_tensor(images)
        
        # Process through conv layers
        x = self.conv_layers(ela_maps)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
