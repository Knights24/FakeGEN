"""
Evaluation script for lightweight Real vs AI detector
Computes detailed metrics: Accuracy, Precision, Recall, F1, ROC-AUC
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, Tuple

from src.models import get_model
from src.data import LightweightRealVsAIDataset, get_transforms


class ModelEvaluator:
    """
    Evaluate trained model on test dataset
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 test_dir: str,
                 device: str = 'cuda',
                 batch_size: int = 32):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            test_dir: Path to test data directory
            device: 'cuda' or 'cpu'
            batch_size: Batch size for evaluation
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        print(f"📊 Loading model from: {checkpoint_path}")
        
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
        print(f"   Best validation accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
        
        # Load test dataset
        print(f"\n📂 Loading test data from: {test_dir}")
        test_transform = get_transforms(augment=False)
        
        self.test_dataset = LightweightRealVsAIDataset(
            data_dir=test_dir,
            transform=test_transform,
            extract_features=self.needs_features
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
    
    @torch.no_grad()
    def predict(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on test set
        
        Returns:
            (true_labels, predictions, probabilities)
        """
        all_labels = []
        all_preds = []
        all_probs = []
        
        print("\n🔍 Running inference...")
        for batch in tqdm(self.test_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].numpy()
            
            if self.needs_features:
                stat_features = batch['stat_features'].to(self.device)
                outputs = self.model(images, stat_features)
            else:
                outputs = self.model(images)
            
            probs = outputs.cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Compute classification metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = 'confusion_matrix.png'):
        """
        Plot and save confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'AI'], 
                   yticklabels=['Real', 'AI'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved confusion matrix to: {save_path}")
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: str = 'roc_curve.png'):
        """
        Plot and save ROC curve
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved ROC curve to: {save_path}")
        plt.close()
    
    def evaluate(self, save_dir: str = 'evaluation_results'):
        """
        Run full evaluation pipeline
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Get predictions
        y_true, y_pred, y_prob = self.predict()
        
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        
        # Print results
        print("\n" + "="*60)
        print("📈 EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"Precision: {metrics['precision']*100:.2f}%")
        print(f"Recall:    {metrics['recall']*100:.2f}%")
        print(f"F1 Score:  {metrics['f1']*100:.2f}%")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print("="*60)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_true, y_pred, save_dir / 'confusion_matrix.png')
        
        # Plot ROC curve
        self.plot_roc_curve(y_true, y_prob, save_dir / 'roc_curve.png')
        
        # Save metrics to file
        metrics_path = save_dir / 'metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write("EVALUATION METRICS\n")
            f.write("="*40 + "\n")
            for metric, value in metrics.items():
                if metric == 'roc_auc':
                    f.write(f"{metric.upper()}: {value:.4f}\n")
                else:
                    f.write(f"{metric.capitalize()}: {value*100:.2f}%\n")
        
        print(f"\n💾 Saved all results to: {save_dir}")
        
        return metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Real vs AI detector')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--test-dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Run evaluation
    evaluator.evaluate(save_dir=args.output_dir)


if __name__ == "__main__":
    main()
