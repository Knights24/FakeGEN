import torch
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    config = {
        'data_root': './datasets/archive/real_vs_fake/real-vs-fake',
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        
        'batch_size': 16,
        'num_epochs': 50,
        'num_workers': 6,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 2,
        
        'model_name': 'efficientnet_b3',
        'pretrained': True,
        'num_classes': 2,
        'img_size': 224,
        
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'scheduler_params': {'T_max': 50, 'eta_min': 1e-6},
        
        'loss_type': 'focal',
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'mixed_precision': True,
        'pin_memory': True,
        'persistent_workers': True,
        
        'save_dir': './checkpoints',
        'save_every': 5,
        'save_best_only': True,
        'early_stopping_patience': 10,
        
        'log_dir': './logs',
        'log_every': 10,
        'use_tensorboard': True,
        'use_wandb': False,
        
        'use_dct': True,
        'dct_block_size': 8,
        'use_srm': True,
        'srm_filters': 5,
        'use_metadata': True,
        'metadata_features': 7,
        'use_ela': True,
        'ela_quality': 90,
        
        'augmentation': {
            'random_crop': True,
            'random_flip': True,
            'color_jitter': True,
            'rotation': 10,
            'normalize': True,
        },
        'seed': 42
    }
    return config


CONFIG = get_config()
BATCH_SIZE = CONFIG['batch_size']
NUM_EPOCHS = CONFIG['num_epochs']
LEARNING_RATE = CONFIG['learning_rate']
DEVICE = CONFIG['device']
IMG_SIZE = CONFIG['img_size']


def print_config():
    print("\n" + "="*70)
    print("  TRAINING CONFIGURATION")
    print("="*70)
    print(f"  GPU: Mixed Precision: {CONFIG['mixed_precision']}")
    print(f"  Workers: {CONFIG['num_workers']}")
    print(f"  Batch: {CONFIG['batch_size']} × {CONFIG['gradient_accumulation_steps']} = {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"  Image Size: {CONFIG['img_size']}×{CONFIG['img_size']}")
    print(f"  Epochs: {CONFIG['num_epochs']}")
    print(f"  Learning Rate: {CONFIG['learning_rate']}")
    print("="*70 + "\n")
