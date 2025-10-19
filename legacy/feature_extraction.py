"""
Feature Extraction Module for Real vs AI Image Detection

Implements:
1. Noise residual extraction (Gaussian high-pass)
2. Pixel correlation analysis (horizontal, vertical, diagonal)
3. DCT frequency domain analysis
4. PRNU fingerprint estimation
5. EXIF metadata extraction
"""

import numpy as np
import cv2
from scipy import signal
from scipy.fftpack import dct, idct
from typing import Dict, Tuple, Optional
import piexif
from PIL import Image


class NoiseExtractor:
    """Extract noise residuals from images"""
    
    def __init__(self, sigma: float = 1.5):
        self.sigma = sigma
    
    def extract_noise_residual(self, image: np.ndarray) -> np.ndarray:
        """
        Extract noise: N(x,y) = I(x,y) - G_σ(I(x,y))
        
        Args:
            image: Input image (H, W, C) or (H, W)
            
        Returns:
            Noise residual map
        """
        if len(image.shape) == 3:
            # Convert to grayscale for noise analysis
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), self.sigma)
        
        # Compute residual
        residual = gray.astype(np.float32) - blurred
        
        return residual
    
    def noise_variance(self, image: np.ndarray) -> float:
        """Compute noise variance"""
        residual = self.extract_noise_residual(image)
        return float(np.var(residual))
    
    def noise_std(self, image: np.ndarray) -> float:
        """Compute noise standard deviation"""
        residual = self.extract_noise_residual(image)
        return float(np.std(residual))


class PixelCorrelationAnalyzer:
    """Analyze pixel correlation patterns"""
    
    def compute_correlation(self, image: np.ndarray, direction: str = 'horizontal') -> float:
        """
        Compute pixel correlation: ρ = Σ(I_i - Ī)(I_j - Ī) / √(Σ(I_i - Ī)²Σ(I_j - Ī)²)
        
        Args:
            image: Input image
            direction: 'horizontal', 'vertical', 'diagonal_main', 'diagonal_anti'
            
        Returns:
            Correlation coefficient
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        gray = gray.astype(np.float32)
        
        if direction == 'horizontal':
            pixels_1 = gray[:, :-1].flatten()
            pixels_2 = gray[:, 1:].flatten()
        elif direction == 'vertical':
            pixels_1 = gray[:-1, :].flatten()
            pixels_2 = gray[1:, :].flatten()
        elif direction == 'diagonal_main':
            pixels_1 = gray[:-1, :-1].flatten()
            pixels_2 = gray[1:, 1:].flatten()
        elif direction == 'diagonal_anti':
            pixels_1 = gray[:-1, 1:].flatten()
            pixels_2 = gray[1:, :-1].flatten()
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        # Compute correlation
        correlation = np.corrcoef(pixels_1, pixels_2)[0, 1]
        
        return float(correlation)
    
    def compute_all_correlations(self, image: np.ndarray) -> Dict[str, float]:
        """Compute correlations in all directions"""
        directions = ['horizontal', 'vertical', 'diagonal_main', 'diagonal_anti']
        correlations = {}
        
        for direction in directions:
            correlations[direction] = self.compute_correlation(image, direction)
        
        # Average correlation
        correlations['average'] = np.mean(list(correlations.values()))
        
        return correlations


class FrequencyAnalyzer:
    """Analyze frequency domain characteristics using DCT"""
    
    def compute_dct(self, image: np.ndarray, block_size: int = 8) -> np.ndarray:
        """
        Compute 2D DCT: F(u,v) = (1/4)C(u)C(v)Σ I(x,y)cos[(2x+1)uπ/2N]cos[(2y+1)vπ/2N]
        
        Args:
            image: Input image
            block_size: Size of DCT blocks
            
        Returns:
            DCT coefficients
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        gray = gray.astype(np.float32)
        
        # Apply DCT block-wise
        h, w = gray.shape
        dct_blocks = []
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                dct_block = cv2.dct(block)
                dct_blocks.append(dct_block)
        
        return np.array(dct_blocks)
    
    def compute_hfer(self, image: np.ndarray, threshold_freq: float = 0.5) -> float:
        """
        Compute High-Frequency Energy Ratio (HFER):
        HFER = Σ_{u,v>f_t} E(u,v) / Σ_{u,v} E(u,v)
        
        Args:
            image: Input image
            threshold_freq: Frequency threshold (0-1)
            
        Returns:
            HFER value
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Compute full-image DCT
        dct_full = cv2.dct(gray.astype(np.float32))
        
        # Compute energy spectrum E(u,v) = |F(u,v)|²
        energy = dct_full ** 2
        
        # Create frequency mask
        h, w = energy.shape
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        
        # Distance from DC component (normalized)
        distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_distance = np.sqrt(center_h**2 + center_w**2)
        normalized_distance = distance / max_distance
        
        # High-frequency mask
        high_freq_mask = normalized_distance > threshold_freq
        
        # Compute HFER
        high_freq_energy = np.sum(energy[high_freq_mask])
        total_energy = np.sum(energy)
        
        hfer = high_freq_energy / (total_energy + 1e-10)
        
        return float(hfer)
    
    def compute_dct_statistics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute various DCT-based statistics"""
        dct_blocks = self.compute_dct(image)
        
        stats = {
            'dct_mean': float(np.mean(np.abs(dct_blocks))),
            'dct_std': float(np.std(dct_blocks)),
            'dct_energy': float(np.sum(dct_blocks ** 2)),
            'hfer': self.compute_hfer(image),
        }
        
        return stats


class PRNUExtractor:
    """Extract PRNU (Photo-Response Non-Uniformity) fingerprint"""
    
    def extract_prnu(self, image: np.ndarray, denoise_sigma: float = 2.0) -> np.ndarray:
        """
        Extract PRNU: K = Σ(W_i * I_i) / Σ(I_i²)
        where W_i = I_i - F(I_i) (noise residual)
        
        Args:
            image: Input image
            denoise_sigma: Denoising filter strength
            
        Returns:
            PRNU noise pattern
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        gray = gray.astype(np.float32) / 255.0
        
        # Denoise with BM3D-style approach (simplified with bilateral filter)
        denoised = cv2.bilateralFilter(gray, 5, denoise_sigma * 10, denoise_sigma * 10)
        
        # Extract noise residual
        noise = gray - denoised
        
        # Normalize
        prnu = noise / (gray + 1e-6)
        
        return prnu
    
    def prnu_statistics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute PRNU-based statistics"""
        prnu = self.extract_prnu(image)
        
        stats = {
            'prnu_mean': float(np.mean(prnu)),
            'prnu_std': float(np.std(prnu)),
            'prnu_energy': float(np.sum(prnu ** 2)),
        }
        
        return stats


class MetadataExtractor:
    """Extract and analyze EXIF metadata"""
    
    def extract_exif(self, image_path: str) -> Dict[str, any]:
        """
        Extract EXIF metadata from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary of EXIF data
        """
        try:
            img = Image.open(image_path)
            exif_dict = piexif.load(img.info.get('exif', b''))
            
            metadata = {
                'has_exif': True,
                'camera_make': None,
                'camera_model': None,
                'software': None,
                'datetime': None,
                'gps': None,
                'iso': None,
                'exposure_time': None,
                'f_number': None,
            }
            
            # Extract common fields
            if '0th' in exif_dict:
                metadata['camera_make'] = exif_dict['0th'].get(piexif.ImageIFD.Make, b'').decode('utf-8', errors='ignore')
                metadata['camera_model'] = exif_dict['0th'].get(piexif.ImageIFD.Model, b'').decode('utf-8', errors='ignore')
                metadata['software'] = exif_dict['0th'].get(piexif.ImageIFD.Software, b'').decode('utf-8', errors='ignore')
                metadata['datetime'] = exif_dict['0th'].get(piexif.ImageIFD.DateTime, b'').decode('utf-8', errors='ignore')
            
            if 'Exif' in exif_dict:
                metadata['iso'] = exif_dict['Exif'].get(piexif.ExifIFD.ISOSpeedRatings)
                metadata['exposure_time'] = exif_dict['Exif'].get(piexif.ExifIFD.ExposureTime)
                metadata['f_number'] = exif_dict['Exif'].get(piexif.ExifIFD.FNumber)
            
            if 'GPS' in exif_dict and exif_dict['GPS']:
                metadata['gps'] = True
            else:
                metadata['gps'] = False
                
        except Exception as e:
            metadata = {'has_exif': False, 'error': str(e)}
        
        return metadata
    
    def metadata_feature_vector(self, image_path: str) -> np.ndarray:
        """
        Convert metadata to binary feature vector
        
        Returns:
            Binary features: [has_exif, has_camera_make, has_camera_model, 
                             has_gps, has_iso, has_exposure, has_f_number]
        """
        metadata = self.extract_exif(image_path)
        
        features = [
            1.0 if metadata.get('has_exif', False) else 0.0,
            1.0 if metadata.get('camera_make') else 0.0,
            1.0 if metadata.get('camera_model') else 0.0,
            1.0 if metadata.get('gps', False) else 0.0,
            1.0 if metadata.get('iso') else 0.0,
            1.0 if metadata.get('exposure_time') else 0.0,
            1.0 if metadata.get('f_number') else 0.0,
        ]
        
        return np.array(features, dtype=np.float32)


class FeatureExtractor:
    """Unified feature extraction interface"""
    
    def __init__(self):
        self.noise_extractor = NoiseExtractor()
        self.correlation_analyzer = PixelCorrelationAnalyzer()
        self.frequency_analyzer = FrequencyAnalyzer()
        self.prnu_extractor = PRNUExtractor()
        self.metadata_extractor = MetadataExtractor()
    
    def extract_all_features(self, image: np.ndarray, image_path: Optional[str] = None) -> Dict[str, any]:
        """
        Extract all statistical features from an image
        
        Args:
            image: Input image array
            image_path: Optional path for metadata extraction
            
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Noise features
        features['noise_variance'] = self.noise_extractor.noise_variance(image)
        features['noise_std'] = self.noise_extractor.noise_std(image)
        
        # Pixel correlation features
        correlations = self.correlation_analyzer.compute_all_correlations(image)
        features.update(correlations)
        
        # Frequency domain features
        dct_stats = self.frequency_analyzer.compute_dct_statistics(image)
        features.update(dct_stats)
        
        # PRNU features
        prnu_stats = self.prnu_extractor.prnu_statistics(image)
        features.update(prnu_stats)
        
        # Metadata features (if path provided)
        if image_path:
            metadata_vec = self.metadata_extractor.metadata_feature_vector(image_path)
            features['metadata_vector'] = metadata_vec
        
        return features
    
    def extract_feature_vector(self, image: np.ndarray, image_path: Optional[str] = None) -> np.ndarray:
        """
        Extract features as a single numerical vector
        
        Returns:
            Feature vector (concatenated statistical features)
        """
        features = self.extract_all_features(image, image_path)
        
        # Build feature vector (excluding metadata_vector which is handled separately)
        feature_list = [
            features['noise_variance'],
            features['noise_std'],
            features['horizontal'],
            features['vertical'],
            features['diagonal_main'],
            features['diagonal_anti'],
            features['average'],
            features['dct_mean'],
            features['dct_std'],
            features['dct_energy'],
            features['hfer'],
            features['prnu_mean'],
            features['prnu_std'],
            features['prnu_energy'],
        ]
        
        feature_vec = np.array(feature_list, dtype=np.float32)
        
        # Append metadata if available
        if 'metadata_vector' in features:
            feature_vec = np.concatenate([feature_vec, features['metadata_vector']])
        
        return feature_vec
