import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import MultiStreamDeepfakeDetector
from src.data import create_dataloaders
from src.training import DeepfakeTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Deepfake Detection Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data_root', 
        type=str, 
        default='./datasets/deepdetect-2025',
        help='Root directory of the dataset'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=16,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=0.01,
        help='Weight decay for AdamW optimizer'
    )
    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=6,
        help='Number of data loading workers'
    )
    
    # Model arguments
    parser.add_argument(
        '--backbone', 
        type=str, 
        default='efficientnet_b3',
        choices=['efficientnet_b0', 'efficientnet_b3', 'efficientnet_b4'],
        help='Backbone architecture'
    )
    parser.add_argument(
        '--pretrained', 
        action='store_true',
        default=True,
        help='Use pretrained backbone weights'
    )
    
    # Loss arguments
    parser.add_argument(
        '--loss', 
        type=str, 
        default='focal',
        choices=['focal', 'cross_entropy'],
        help='Loss function'
    )
    parser.add_argument(
        '--focal_alpha', 
        type=float, 
        default=0.25,
        help='Alpha parameter for focal loss'
    )
    parser.add_argument(
        '--focal_gamma', 
        type=float, 
        default=2.0,
        help='Gamma parameter for focal loss'
    )
    
    # Output arguments
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='./checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--save_every', 
        type=int, 
        default=5,
        help='Save checkpoint every N epochs'
    )
    
    # Early stopping
    parser.add_argument(
        '--patience', 
        type=int, 
        default=10,
        help='Early stopping patience'
    )
    
    # Misc
    parser.add_argument(
        '--img_size', 
        type=int, 
        default=224,
        help='Input image size'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Print configuration
    print("="*60)
    print("DEEPFAKE DETECTION MODEL TRAINING")
    print("="*60)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Backbone: {args.backbone}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Loss Function: {args.loss}")
    print(f"Data Root: {args.data_root}")
    print("="*60)
    
    # Create dataloaders
    print("\n[1/4] Loading datasets...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            root_dir=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            include_metadata=True,
            pin_memory=(device == 'cuda')
        )
        print(f"✓ Train: {len(train_loader.dataset)} samples")
        print(f"✓ Val: {len(val_loader.dataset)} samples")
        print(f"✓ Test: {len(test_loader.dataset)} samples")
    except Exception as e:
        print(f"✗ Error loading datasets: {e}")
        print("\nPlease ensure your dataset is organized as:")
        print("  datasets/")
        print("  ├── train/")
        print("  │   ├── real/")
        print("  │   └── fake/")
        print("  ├── val/")
        print("  │   ├── real/")
        print("  │   └── fake/")
        print("  └── test/")
        print("      ├── real/")
        print("      └── fake/")
        sys.exit(1)
    
    # Initialize model
    print("\n[2/4] Initializing model...")
    model = MultiStreamDeepfakeDetector(
        num_classes=2,
        pretrained=args.pretrained,
        backbone=args.backbone
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    print("\n[3/4] Initializing trainer...")
    trainer = DeepfakeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )
    print(f"✓ Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
    print(f"✓ Scheduler: CosineAnnealingLR")
    print(f"✓ Loss: {args.loss.capitalize()}")
    
    # Train
    print("\n[4/4] Starting training...")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"✓ Resumed from epoch {start_epoch}")
    
    history = trainer.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        early_stopping_patience=args.patience,
        save_every=args.save_every,
        start_epoch=start_epoch
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    test_metrics = trainer.evaluate(test_loader)
    
    # Save final results
    results_path = os.path.join(args.save_dir, 'training_results.txt')
    with open(results_path, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Best F1: {history['best_f1']:.4f} (Epoch {history['best_epoch']})\n")
        f.write(f"Total Time: {history['total_time']/3600:.2f} hours\n")
        f.write("\nTest Metrics:\n")
        for k, v in test_metrics.items():
            if k != 'confusion_matrix':
                f.write(f"  {k}: {v:.4f}\n")
    
    print(f"\n✓ Results saved to {results_path}")
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
