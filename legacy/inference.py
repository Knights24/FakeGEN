"""
Inference script for lightweight Real vs AI detector
Predict on single images or batches
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, List, Dict
import time

from lightweight_model import get_model
from lightweight_dataset import get_transforms
from simple_features import SimpleFeaturesExtractor


class RealVsAIPredictor:
    """
    Easy-to-use predictor for Real vs AI image detection
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"🔮 Loading Real vs AI Detector...")
        print(f"   Device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_type = checkpoint['model_type']
        
        # Build model
        if model_type == 'efficientnet':
            self.model = get_model('efficientnet', pretrained=False)
            self.needs_features = False
        else:
            self.model = get_model(model_type, pretrained=False)
            self.needs_features = True
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Loaded {model_type} model")
        
        # Transforms
        self.transform = get_transforms(augment=False)
        
        # Feature extractor
        if self.needs_features:
            self.feature_extractor = SimpleFeaturesExtractor()
    
    @torch.no_grad()
    def predict_single(self, image_path: str) -> Dict[str, float]:
        """
        Predict on a single image
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dict with 'probability' (AI confidence) and 'prediction' (Real/AI)
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features if needed
        if self.needs_features:
            stat_features = self.feature_extractor.extract_all(
                self.transform(image), 
                image_path
            ).unsqueeze(0).to(self.device)
            output = self.model(image_tensor, stat_features)
        else:
            output = self.model(image_tensor)
        
        probability = output.item()
        prediction = 'AI' if probability > 0.5 else 'Real'
        
        return {
            'probability': probability,
            'prediction': prediction,
            'confidence': max(probability, 1 - probability)
        }
    
    @torch.no_grad()
    def predict_batch(self, image_paths: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for inference
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            batch_features = []
            
            # Load batch
            for path in batch_paths:
                image = Image.open(path).convert('RGB')
                image_tensor = self.transform(image)
                batch_images.append(image_tensor)
                
                if self.needs_features:
                    feat = self.feature_extractor.extract_all(image_tensor, path)
                    batch_features.append(feat)
            
            # Stack batch
            batch_images = torch.stack(batch_images).to(self.device)
            
            # Predict
            if self.needs_features:
                batch_features = torch.stack(batch_features).to(self.device)
                outputs = self.model(batch_images, batch_features)
            else:
                outputs = self.model(batch_images)
            
            # Process results
            probabilities = outputs.cpu().numpy().flatten()
            for prob in probabilities:
                results.append({
                    'probability': float(prob),
                    'prediction': 'AI' if prob > 0.5 else 'Real',
                    'confidence': float(max(prob, 1 - prob))
                })
        
        return results
    
    def predict_folder(self, folder_path: str, pattern: str = '*.jpg') -> Dict[str, Dict]:
        """
        Predict on all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            pattern: File pattern (default: *.jpg)
        
        Returns:
            Dict mapping filename to prediction
        """
        folder = Path(folder_path)
        image_paths = list(folder.glob(pattern))
        
        if not image_paths:
            print(f"⚠️  No images found in {folder_path} with pattern {pattern}")
            return {}
        
        print(f"🔍 Predicting on {len(image_paths)} images...")
        
        start_time = time.time()
        results = self.predict_batch([str(p) for p in image_paths])
        elapsed = time.time() - start_time
        
        print(f"✅ Completed in {elapsed:.2f}s ({len(image_paths)/elapsed:.1f} images/sec)")
        
        # Map results to filenames
        output = {}
        for path, result in zip(image_paths, results):
            output[path.name] = result
        
        return output


def print_prediction(result: Dict[str, float], image_path: str = None):
    """Pretty print prediction result"""
    if image_path:
        print(f"\n{'='*60}")
        print(f"Image: {image_path}")
    
    print(f"{'='*60}")
    prediction = result['prediction']
    probability = result['probability']
    confidence = result['confidence']
    
    # Color coding
    if prediction == 'Real':
        emoji = "📷"
        bar_char = "█"
    else:
        emoji = "🤖"
        bar_char = "▓"
    
    # Progress bar
    bar_length = 40
    filled = int(confidence * bar_length)
    bar = bar_char * filled + "░" * (bar_length - filled)
    
    print(f"{emoji} Prediction: {prediction}")
    print(f"   AI Probability: {probability:.2%}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   [{bar}]")
    print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict Real vs AI on images')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str,
                       help='Path to single image')
    parser.add_argument('--folder', type=str,
                       help='Path to folder of images')
    parser.add_argument('--pattern', type=str, default='*.jpg',
                       help='File pattern for folder prediction')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--output', type=str,
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = RealVsAIPredictor(args.checkpoint, args.device)
    
    # Single image
    if args.image:
        result = predictor.predict_single(args.image)
        print_prediction(result, args.image)
    
    # Folder of images
    elif args.folder:
        results = predictor.predict_folder(args.folder, args.pattern)
        
        # Print summary
        ai_count = sum(1 for r in results.values() if r['prediction'] == 'AI')
        real_count = len(results) - ai_count
        
        print(f"\n📊 Summary:")
        print(f"   Total images: {len(results)}")
        print(f"   Real: {real_count}")
        print(f"   AI: {ai_count}")
        
        # Save to JSON if specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Saved results to: {args.output}")
        
        # Print top 5 most confident AI predictions
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['probability'], 
                              reverse=True)
        
        print(f"\n🤖 Top 5 AI predictions:")
        for filename, result in sorted_results[:5]:
            print(f"   {filename}: {result['probability']:.2%}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
