import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import MultiStreamDeepfakeDetector
from src.data import get_transforms
from src.features import MetadataAnalyzer


class DeepfakeInference:
    """
    Inference pipeline for trained deepfake detection model.
    
    Args:
        model_path: Path to trained model checkpoint
        device: Device to run inference on ('cuda' or 'cpu')
        backbone: Model backbone architecture
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        backbone: str = 'efficientnet_b3'
    ):
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize model
        self.model = MultiStreamDeepfakeDetector(
            num_classes=2,
            pretrained=False,
            backbone=backbone
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get checkpoint info
        self.checkpoint_epoch = checkpoint.get('epoch', 'unknown')
        self.checkpoint_metrics = checkpoint.get('metrics', {})
        
        # Initialize transforms and metadata analyzer
        self.transform = get_transforms('test', img_size=224)
        self.metadata_analyzer = MetadataAnalyzer()
        
        print(f"✓ Loaded model from epoch {self.checkpoint_epoch}")
        print(f"  Device: {self.device}")
        if self.checkpoint_metrics:
            print(f"  Checkpoint F1: {self.checkpoint_metrics.get('f1', 'N/A'):.4f}")
    
    @torch.no_grad()
    def predict(self, image_path: str) -> Dict:
        """
        Predict if a single image is a deepfake.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            result: Dictionary with prediction, confidence, and probabilities
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Apply transforms
        if self.transform is not None:
            try:
                # Albumentations
                augmented = self.transform(image=image_np)
                image_tensor = augmented['image']
            except:
                # Fallback to basic transform
                image = image.resize((224, 224))
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        else:
            image = image.resize((224, 224))
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        
        # Extract metadata
        metadata = self.metadata_analyzer.extract_features(image_path)
        metadata_tensor = torch.tensor([
            metadata['has_exif'],
            metadata['camera_make'],
            metadata['camera_model'],
            metadata['software'],
            metadata['gps_info'],
            metadata['color_space'],
            metadata['compression']
        ], dtype=torch.float32)
        
        # Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        metadata_tensor = metadata_tensor.unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(image_tensor, metadata_tensor)
        probs = F.softmax(outputs, dim=1)
        
        # Get prediction
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
        
        result = {
            'prediction': 'FAKE' if pred_class == 1 else 'REAL',
            'label': pred_class,
            'confidence': confidence * 100,
            'fake_probability': probs[0, 1].item() * 100,
            'real_probability': probs[0, 0].item() * 100,
            'metadata_score': metadata['authenticity_score'],
            'has_exif': bool(metadata['has_exif']),
            'has_camera_info': bool(metadata['camera_make'] or metadata['camera_model'])
        }
        
        return result
    
    @torch.no_grad()
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            results: List of prediction dictionaries
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                result['path'] = img_path
                results.append(result)
            except Exception as e:
                results.append({
                    'path': img_path,
                    'error': str(e),
                    'prediction': 'ERROR'
                })
        
        return results
    
    @torch.no_grad()
    def predict_with_explanation(self, image_path: str) -> Dict:
        """
        Predict with detailed explanation including feature weights.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            result: Detailed prediction with explanations
        """
        # Get basic prediction
        result = self.predict(image_path)
        
        # Get fusion weights
        if hasattr(self.model, 'get_fusion_weights'):
            weights = self.model.get_fusion_weights()
            result['fusion_weights'] = weights
            
            # Interpretation
            explanations = []
            
            # Check metadata
            if not result['has_exif']:
                explanations.append("No EXIF metadata found (common in AI-generated images)")
            if not result['has_camera_info']:
                explanations.append("No camera make/model information")
            
            # Add confidence interpretation
            if result['fake_probability'] > 90:
                explanations.append("Very high confidence this is AI-generated")
            elif result['fake_probability'] > 70:
                explanations.append("Likely AI-generated")
            elif result['fake_probability'] > 50:
                explanations.append("Possibly AI-generated")
            elif result['real_probability'] > 90:
                explanations.append("Very high confidence this is authentic")
            
            result['explanations'] = explanations
        
        return result
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'device': self.device,
            'checkpoint_epoch': self.checkpoint_epoch,
            'checkpoint_metrics': self.checkpoint_metrics,
            'feature_dims': self.model.get_feature_dims() if hasattr(self.model, 'get_feature_dims') else {},
            'fusion_weights': self.model.get_fusion_weights() if hasattr(self.model, 'get_fusion_weights') else {}
        }


def main():
    """Demo usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, default='./checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    # Initialize detector
    print("Loading model...")
    detector = DeepfakeInference(args.model, device=args.device)
    
    # Predict
    print(f"\nAnalyzing: {args.image}")
    result = detector.predict_with_explanation(args.image)
    
    # Print results
    print("\n" + "="*50)
    print("DEEPFAKE DETECTION RESULT")
    print("="*50)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Fake Probability: {result['fake_probability']:.2f}%")
    print(f"Real Probability: {result['real_probability']:.2f}%")
    print(f"\nMetadata Analysis:")
    print(f"  Has EXIF: {result['has_exif']}")
    print(f"  Has Camera Info: {result['has_camera_info']}")
    
    if 'explanations' in result:
        print(f"\nExplanations:")
        for exp in result['explanations']:
            print(f"  • {exp}")
    
    if 'fusion_weights' in result:
        print(f"\nModel Fusion Weights:")
        for k, v in result['fusion_weights'].items():
            print(f"  {k}: {v:.4f}")
    
    print("="*50)


if __name__ == "__main__":
    main()
