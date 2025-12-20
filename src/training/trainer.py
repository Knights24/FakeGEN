import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DeepfakeTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str = 'cuda',
                 learning_rate: float = 1e-4, weight_decay: float = 0.01, loss_type: str = 'focal',
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0, accumulation_steps: int = 1):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accumulation_steps = accumulation_steps
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if loss_type == 'focal':
            self.criterion = self._focal_loss
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.scaler = GradScaler('cuda')
        self.use_amp = (device == 'cuda')
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_metrics: List[Dict] = []
        self.best_f1 = 0.0
        self.best_epoch = 0
    
    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - p_t) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            metadata = batch['metadata'].to(self.device)
            labels = batch['label'].to(self.device)
            
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images, metadata)
                    loss = self.criterion(outputs, labels) / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images, metadata)
                loss = self.criterion(outputs, labels) / self.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            running_loss += loss.item() * self.accumulation_steps
            
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch} [{batch_idx}/{num_batches}] Loss: {loss.item() * self.accumulation_steps:.4f} LR: {current_lr:.2e}")
        
        return running_loss / num_batches
    
    @torch.no_grad()
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            metadata = batch['metadata'].to(self.device)
            labels = batch['label'].to(self.device)
            
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images, metadata)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images, metadata)
                loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
        
        avg_loss = running_loss / len(self.val_loader)
        
        if SKLEARN_AVAILABLE:
            metrics = {
                'accuracy': accuracy_score(all_labels, all_preds),
                'precision': precision_score(all_labels, all_preds, zero_division=0),
                'recall': recall_score(all_labels, all_preds, zero_division=0),
                'f1': f1_score(all_labels, all_preds, zero_division=0),
                'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
            }
        else:
            correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
            metrics = {'accuracy': correct / len(all_labels), 'f1': 0.0}
        
        return avg_loss, metrics
    
    def train(self, num_epochs: int, save_dir: str = './checkpoints', early_stopping_patience: int = 10):
        os.makedirs(save_dir, exist_ok=True)
        best_f1 = 0.0
        patience_counter = 0
        
        self.scheduler.T_max = num_epochs
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Accumulation Steps: {self.accumulation_steps}")
        print("-" * 70)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate_epoch()
            
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            epoch_time = time.time() - epoch_start
            
            print(f"\nEpoch {epoch}/{num_epochs} - Time: {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Metrics: Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}")
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.best_f1 = best_f1
                self.best_epoch = epoch
                patience_counter = 0
                
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                self.save_checkpoint(checkpoint_path, epoch, val_metrics)
                print(f"  ✓ New best model saved (F1: {best_f1:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            print("-" * 70)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best F1: {self.best_f1:.4f} at epoch {self.best_epoch}")
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }, path)
    
    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        return checkpoint['epoch']
