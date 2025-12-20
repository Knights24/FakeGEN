import os
import numpy as np
import torch
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from typing import Dict, List, Optional, Union


class MetadataAnalyzer:
    """
    Extract and analyze image metadata for deepfake detection.
    
    Authentic images contain rich metadata from camera sensors,
    while AI-generated content typically lacks these characteristics.
    
    Features extracted:
    - EXIF presence and completeness
    - Camera hardware information
    - GPS and location data
    - Software editing history
    - Color space and compression info
    """
    
    # Define the feature names for consistent ordering
    FEATURE_NAMES = [
        'has_exif',
        'camera_make',
        'camera_model',
        'software',
        'gps_info',
        'color_space',
        'compression'
    ]
    
    NUM_FEATURES = len(FEATURE_NAMES)
    
    def __init__(self):
        """Initialize the metadata analyzer."""
        pass
    
    def extract_features(self, image_path: str) -> Dict[str, float]:
        """
        Extract all metadata features from image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            features: Dictionary with 7 binary features + authenticity score
        """
        features = {
            'has_exif': 0.0,
            'camera_make': 0.0,
            'camera_model': 0.0,
            'software': 0.0,
            'gps_info': 0.0,
            'color_space': 0.0,
            'compression': 0.0,
            'authenticity_score': 0.0
        }
        
        try:
            # Open image
            img = Image.open(image_path)
            
            # Get EXIF data using getexif() method (PIL 6.0+)
            exif_data = img.getexif()
            
            if exif_data:
                features['has_exif'] = 1.0
                
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, str(tag_id))
                    
                    if tag == 'Make':
                        features['camera_make'] = 1.0
                    elif tag == 'Model':
                        features['camera_model'] = 1.0
                    elif tag == 'Software':
                        features['software'] = 1.0
                    elif tag == 'GPSInfo':
                        features['gps_info'] = 1.0
                    elif tag == 'ColorSpace':
                        features['color_space'] = 1.0
                    elif tag == 'Compression':
                        features['compression'] = 1.0
            
            # Try alternative EXIF extraction for some image formats
            if not features['has_exif']:
                try:
                    # Some formats store EXIF differently
                    info = img.info
                    if 'exif' in info or 'icc_profile' in info:
                        features['has_exif'] = 1.0
                except:
                    pass
            
            # Calculate authenticity score based on camera-specific features
            # AI-generated images often lack camera make, model, and GPS
            camera_features = (
                features['camera_make'] + 
                features['camera_model'] + 
                features['gps_info']
            )
            features['authenticity_score'] = camera_features / 3.0
            
        except Exception as e:
            # Log error but don't crash - return zero features
            print(f"Warning: Error extracting metadata from {image_path}: {e}")
        
        return features
    
    def extract_features_tensor(self, image_path: str) -> torch.Tensor:
        """
        Extract metadata features and return as tensor.
        
        Args:
            image_path: Path to image file
            
        Returns:
            tensor: [7] tensor with binary features (excluding authenticity_score)
        """
        features_dict = self.extract_features(image_path)
        
        feature_vector = [
            features_dict['has_exif'],
            features_dict['camera_make'],
            features_dict['camera_model'],
            features_dict['software'],
            features_dict['gps_info'],
            features_dict['color_space'],
            features_dict['compression']
        ]
        
        return torch.tensor(feature_vector, dtype=torch.float32)
    
    def extract_detailed_metadata(self, image_path: str) -> Dict:
        """
        Extract detailed metadata for analysis (not for model input).
        
        Args:
            image_path: Path to image file
            
        Returns:
            metadata: Dictionary with all available metadata
        """
        metadata = {
            'basic': {},
            'exif': {},
            'gps': {},
            'camera': {},
            'software': {},
        }
        
        try:
            img = Image.open(image_path)
            
            # Basic image info
            metadata['basic'] = {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'filename': os.path.basename(image_path),
            }
            
            # EXIF data
            exif_data = img.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, str(tag_id))
                    
                    # Store in appropriate category
                    if tag in ['Make', 'Model', 'LensModel', 'LensMake']:
                        metadata['camera'][tag] = str(value)
                    elif tag == 'GPSInfo':
                        metadata['gps']['has_gps'] = True
                        # Parse GPS data if available
                        try:
                            for gps_tag_id, gps_value in value.items():
                                gps_tag = GPSTAGS.get(gps_tag_id, str(gps_tag_id))
                                metadata['gps'][gps_tag] = str(gps_value)
                        except:
                            pass
                    elif tag in ['Software', 'ProcessingSoftware']:
                        metadata['software'][tag] = str(value)
                    else:
                        metadata['exif'][tag] = str(value)
                        
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    def batch_extract(self, image_paths: List[str]) -> torch.Tensor:
        """
        Extract metadata features for a batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            features: [B, 7] tensor with metadata features
        """
        features_list = []
        for path in image_paths:
            features_list.append(self.extract_features_tensor(path))
        
        return torch.stack(features_list)
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get the list of feature names in order."""
        return MetadataAnalyzer.FEATURE_NAMES.copy()
    
    def is_likely_authentic(self, image_path: str, threshold: float = 0.5) -> bool:
        """
        Quick check if image is likely authentic based on metadata.
        
        Args:
            image_path: Path to image file
            threshold: Authenticity score threshold (default=0.5)
            
        Returns:
            is_authentic: True if image appears to have camera metadata
        """
        features = self.extract_features(image_path)
        return features['authenticity_score'] >= threshold
