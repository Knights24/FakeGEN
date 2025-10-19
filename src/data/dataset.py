"""
Lightweight Dataset Loader for Real vs AI Detection
Optimized for speed and memory efficiency
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from src.features import SimpleFeaturesExtractor
from itertools import chain


class LightweightRealVsAIDataset(Dataset):
    """
    Dataset for Real vs AI image classification
    
    Expected directory structure:
    data/
        real/
            img1.jpg
            img2.jpg
            ...
        ai/
            img1.jpg
            img2.jpg
            ...
    """
    
    def __init__(self, 
                 data_dir: str,
                 transform: Optional[T.Compose] = None,
                 extract_features: bool = True,
                 max_samples: Optional[int] = None,
                 real_subdir: Optional[str] = None,
                 ai_subdir: Optional[str] = None):
        """
        Args:
            data_dir: Path to data directory containing class subfolders
            transform: torchvision transforms
            extract_features: Whether to extract statistical features
            max_samples: Limit number of samples (useful for quick testing)
            real_subdir: Optional explicit name for the REAL class folder
            ai_subdir: Optional explicit name for the AI class folder
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.extract_features = extract_features
        
        if extract_features:
            self.feature_extractor = SimpleFeaturesExtractor()
        
        # Supported folder name candidates
        real_candidates = [
            'real', 'final_real', 'real_images', 'camera', 'photos',
            'real_humans', 'real_humans_original'
        ]
        ai_candidates = [
            'ai', 'final_ai', 'synthetic', 'ai_images', 'deepfake', 'fake', 'generated'
        ]

        # If user provided explicit subdir names, use them first
        if real_subdir:
            real_candidates = [real_subdir] + real_candidates
        if ai_subdir:
            ai_candidates = [ai_subdir] + ai_candidates

        # Resolve class directories
        def resolve_subdir(base: Path, names: List[str]) -> Optional[Path]:
            for name in names:
                p = base / name
                if p.exists() and p.is_dir():
                    return p
            return None

        real_dir = resolve_subdir(self.data_dir, real_candidates)
        ai_dir = resolve_subdir(self.data_dir, ai_candidates)

        # Load file paths and labels
        self.samples = []
        self.labels = []
        
        # Helper to collect image files recursively
        supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        def collect_images(folder: Path) -> List[Path]:
            return [p for p in folder.rglob('*') if p.suffix.lower() in supported_exts]

        # Real images (label = 0)
        if real_dir is not None:
            real_files = collect_images(real_dir)
            if max_samples:
                real_files = real_files[:max_samples]
            for img_path in real_files:
                self.samples.append(str(img_path))
                self.labels.append(0)
        
        # AI images (label = 1)
        if ai_dir is not None:
            ai_files = collect_images(ai_dir)
            if max_samples:
                ai_files = ai_files[:max_samples]
            for img_path in ai_files:
                self.samples.append(str(img_path))
                self.labels.append(1)
        
        if len(self.samples) == 0:
            raise ValueError(
                f"No images found in {data_dir}. Expected class subdirectories. "
                f"Tried real folders: {real_candidates}, ai folders: {ai_candidates}"
            )
        
        print(f"Loaded {len(self.samples)} images from '{self.data_dir}':")
        if real_dir is not None:
            print(f"  - Real [{real_dir.name}]: {self.labels.count(0)}")
        if ai_dir is not None:
            print(f"  - AI   [{ai_dir.name}]: {self.labels.count(1)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
        - 'image': [3, 224, 224] tensor
        - 'label': 0 (real) or 1 (AI)
        - 'stat_features': [6] tensor (if extract_features=True)
        """
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)
        
        result = {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.float32)
        }
        
        # Extract statistical features if needed
        if self.extract_features:
            stat_features = self.feature_extractor.extract_all(image_tensor, img_path)
            result['stat_features'] = stat_features
        
        return result
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training"""
        real_count = self.labels.count(0)
        ai_count = self.labels.count(1)
        total = len(self.labels)
        
        weights = torch.tensor([
            total / (2 * real_count),  # Weight for real
            total / (2 * ai_count)      # Weight for AI
        ])
        
        return weights


def get_transforms(img_size: int = 224, 
                   augment: bool = True,
                   normalize: bool = True) -> T.Compose:
    """
    Get image transforms
    
    Args:
        img_size: Target image size (default: 224 for EfficientNet)
        augment: Apply data augmentation
        normalize: Apply normalization
    
    Returns:
        torchvision.transforms.Compose
    """
    transforms = []
    
    # Resize
    transforms.append(T.Resize((img_size, img_size)))
    
    # Augmentation for training
    if augment:
        transforms.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])
    
    # Convert to tensor
    transforms.append(T.ToTensor())
    
    # Normalization (use simpler [-1, 1] range or ImageNet stats)
    if normalize:
        # Simple normalization: [0, 1] -> [-1, 1]
        transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    
    return T.Compose(transforms)


def create_balanced_sampler(dataset: LightweightRealVsAIDataset) -> WeightedRandomSampler:
    """
    Create a balanced sampler to handle class imbalance
    """
    class_weights = dataset.get_class_weights()
    sample_weights = [class_weights[label] for label in dataset.labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def get_dataloaders(train_dir: str,
                   val_dir: Optional[str] = None,
                   batch_size: int = 32,
                   num_workers: int = 2,
                   img_size: int = 224,
                   extract_features: bool = True,
                   use_balanced_sampler: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders
    
    Args:
        train_dir: Path to training data
        val_dir: Path to validation data (optional)
        batch_size: Batch size
        num_workers: Number of worker threads (2 recommended for RTX 4060)
        img_size: Image size (224 for EfficientNet)
        extract_features: Extract statistical features
        use_balanced_sampler: Use balanced sampling for imbalanced datasets
    
    Returns:
        (train_loader, val_loader)
    """
    # Training transforms (with augmentation)
    train_transform = get_transforms(img_size=img_size, augment=True)
    
    # Validation transforms (no augmentation)
    val_transform = get_transforms(img_size=img_size, augment=False)
    
    # Training dataset
    train_dataset = LightweightRealVsAIDataset(
        data_dir=train_dir,
        transform=train_transform,
        extract_features=extract_features
    )
    
    # Create train loader
    if use_balanced_sampler:
        sampler = create_balanced_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    # Validation loader
    val_loader = None
    if val_dir and os.path.exists(val_dir):
        val_dataset = LightweightRealVsAIDataset(
            data_dir=val_dir,
            transform=val_transform,
            extract_features=extract_features
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing Lightweight Dataset...")
    
    # Test transforms
    print("\n1. Testing transforms:")
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    print(f"   Train transform: {len(train_transform.transforms)} steps")
    print(f"   Val transform: {len(val_transform.transforms)} steps")
    
    # Test with dummy images
    dummy_img = Image.new('RGB', (512, 512), color='red')
    transformed = train_transform(dummy_img)
    print(f"   Transformed image shape: {transformed.shape}")
    print(f"   Value range: [{transformed.min():.2f}, {transformed.max():.2f}]")
    
    # If data directory exists, test dataset
    if os.path.exists('data'):
        print("\n2. Testing dataset loading:")
        try:
            # Try to load a small sample
            dataset = LightweightRealVsAIDataset(
                data_dir='data',
                transform=train_transform,
                extract_features=True,
                max_samples=10
            )
            
            print(f"   Dataset size: {len(dataset)}")
            
            # Get one sample
            sample = dataset[0]
            print(f"   Sample keys: {sample.keys()}")
            print(f"   Image shape: {sample['image'].shape}")
            print(f"   Label: {sample['label'].item()}")
            if 'stat_features' in sample:
                print(f"   Statistical features shape: {sample['stat_features'].shape}")
            
            # Test dataloader
            print("\n3. Testing dataloader:")
            train_loader, _ = get_dataloaders(
                train_dir='data',
                batch_size=4,
                num_workers=0,  # Use 0 for testing
                extract_features=True
            )
            
            batch = next(iter(train_loader))
            print(f"   Batch image shape: {batch['image'].shape}")
            print(f"   Batch labels shape: {batch['label'].shape}")
            if 'stat_features' in batch:
                print(f"   Batch features shape: {batch['stat_features'].shape}")
            
        except ValueError as e:
            print(f"   No data found: {e}")
    else:
        print("\n2. No 'data' directory found. Skipping dataset test.")
        print("   Create data/real/ and data/ai/ directories with images to test.")
    
    print("\n✅ Dataset module working!")
