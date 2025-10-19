"""
Lightweight Training Script for Real vs AI Detection
Optimized for RTX 4060 Laptop GPU with:
- Mixed precision training (AMP)
- Memory management
- Checkpoint saving
- Progress tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
import json

from lightweight_model import get_model
from lightweight_dataset import get_dataloaders


class LightweightTrainer:
    """
    Trainer for lightweight Real vs AI detection models
    """
    
    def __init__(self,
                 model_type: str = 'hybrid',
                 train_dir: str = 'data/train',
                 val_dir: Optional[str] = 'data/val',
                 batch_size: int = 32,
                 num_epochs: int = 50,
                 learning_rate: float = 1e-4,
                 device: str = 'cuda',
                 checkpoint_dir: str = 'checkpoints',
                 use_amp: bool = True,
                 num_workers: int = 2):
        """
        Args:
            model_type: 'hybrid', 'efficientnet', or 'tiny'
            train_dir: Path to training data
            val_dir: Path to validation data
            batch_size: Batch size (32 recommended for RTX 4060)
            num_epochs: Number of training epochs
            learning_rate: Learning rate (1e-4 works well)
            device: 'cuda' or 'cpu'
            checkpoint_dir: Directory to save checkpoints
            use_amp: Use automatic mixed precision (saves ~30% VRAM)
            num_workers: DataLoader workers (2 recommended)
        """
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.use_amp = use_amp and torch.cuda.is_available()
        
        print(f"🚀 Initializing Lightweight Trainer")
        print(f"   Model: {model_type}")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {batch_size}")
        print(f"   Mixed precision: {self.use_amp}")
        
        # Memory optimization for RTX 4060
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.backends.cudnn.benchmark = True
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM limit: 80%")
        
        # Build model
        if model_type == 'efficientnet':
            self.model = get_model('efficientnet', pretrained=True, dropout=0.3)
            self.needs_features = False
        else:
            self.model = get_model(model_type, pretrained=True, dropout=0.3)
            self.needs_features = True
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        if hasattr(self.model, 'get_num_params'):
            print(f"   Parameters: {self.model.get_num_params():,}")
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler (cosine annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Load dataloaders
        print(f"\n📂 Loading data...")
        self.train_loader, self.val_loader = get_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            extract_features=self.needs_features
        )
        
        print(f"   Training batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"   Validation batches: {len(self.val_loader)}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0.0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device).unsqueeze(1)
            
            # Get statistical features if needed
            if self.needs_features:
                stat_features = batch['stat_features'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    if self.needs_features:
                        outputs = self.model(images, stat_features)
                    else:
                        outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Normal training
                if self.needs_features:
                    outputs = self.model(images, stat_features)
                else:
                    outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return {'loss': epoch_loss, 'acc': epoch_acc}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {'loss': 0.0, 'acc': 0.0}
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device).unsqueeze(1)
            
            if self.needs_features:
                stat_features = batch['stat_features'].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    if self.needs_features:
                        outputs = self.model(images, stat_features)
                    else:
                        outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                if self.needs_features:
                    outputs = self.model(images, stat_features)
                else:
                    outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100 * correct / total
        
        return {'loss': val_loss, 'acc': val_acc}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'model_type': self.model_type
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'{self.model_type}_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f'{self.model_type}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"   💾 Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    
    def train(self):
        """Full training loop"""
        print(f"\n🏋️ Starting training for {self.num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['learning_rate'].append(current_lr)
            
            # Print summary
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['acc']:.2f}%")
            print(f"   Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.2f}%")
            print(f"   Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_metrics['acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['acc']
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"   Total time: {elapsed_time/60:.1f} minutes")
        print(f"   Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}")
        
        # Save final history
        history_path = self.checkpoint_dir / f'{self.model_type}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train lightweight Real vs AI detector')
    parser.add_argument('--model', type=str, default='hybrid',
                       choices=['hybrid', 'efficientnet', 'tiny'],
                       help='Model type')
    parser.add_argument('--train-dir', type=str, default='data/train',
                       help='Training data directory')
    parser.add_argument('--val-dir', type=str, default='data/val',
                       help='Validation data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = LightweightTrainer(
        model_type=args.model,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.no_amp,
        num_workers=args.workers
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
