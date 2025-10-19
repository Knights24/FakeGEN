"""
Dataset and DataLoader utilities for Real vs AI Image Detection
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, Optional, List
from feature_extraction import FeatureExtractor, NoiseExtractor


class RealVsAIDataset(Dataset):
    """
    Dataset for Real vs AI image classification
    
    Directory structure:
        data/
            real/
                image1.jpg
                image2.jpg
                ...
            ai/
                image1.jpg
                image2.jpg
                ...
    """
    
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None,
                 extract_features: bool = False, extract_noise: bool = False):
        """
        Args:
            root_dir: Root directory containing 'real' and 'ai' folders
            transform: Image transformations
            extract_features: Whether to extract statistical features
            extract_noise: Whether to extract noise residual maps
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extract_features = extract_features
        self.extract_noise = extract_noise
        
        # Feature extractors
        if extract_features:
            self.feature_extractor = FeatureExtractor()
        if extract_noise:
            self.noise_extractor = NoiseExtractor()
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        
        # Real images (label = 0)
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(real_dir, img_name))
                    self.labels.append(0)
        
        # AI images (label = 1)
        ai_dir = os.path.join(root_dir, 'ai')
        if os.path.exists(ai_dir):
            for img_name in os.listdir(ai_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(ai_dir, img_name))
                    self.labels.append(1)
        
        print(f"Loaded {len(self.labels)} images: "
              f"{sum([1 for l in self.labels if l == 0])} real, "
              f"{sum([1 for l in self.labels if l == 1])} AI")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # Apply transforms
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        
        # Prepare return data
        data = {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
        }
        
        # Extract statistical features if needed
        if self.extract_features:
            features = self.feature_extractor.extract_feature_vector(image_np, img_path)
            data['stat_features'] = torch.from_numpy(features[:14])  # Statistical features
            data['meta_features'] = torch.from_numpy(features[14:21])  # Metadata features
        
        # Extract noise residual if needed
        if self.extract_noise:
            noise = self.noise_extractor.extract_noise_residual(image_np)
            # Normalize and convert to tensor
            noise = (noise - noise.mean()) / (noise.std() + 1e-8)
            noise_tensor = torch.from_numpy(noise).unsqueeze(0).float()
            data['noise_map'] = noise_tensor
        
        return data


def get_transforms(img_size: int = 224, augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and validation transforms
    
    Args:
        img_size: Target image size
        augment: Whether to apply data augmentation
    
    Returns:
        (train_transform, val_transform)
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_dataloaders(train_dir: str, val_dir: str, batch_size: int = 32,
                   num_workers: int = 4, img_size: int = 224,
                   model_type: str = 'cnn') -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Image size
        model_type: Model type ('cnn', 'hybrid', 'noise_cnn')
    
    Returns:
        (train_loader, val_loader)
    """
    # Get transforms
    train_transform, val_transform = get_transforms(img_size, augment=True)
    
    # Determine what features to extract based on model type
    extract_features = (model_type == 'hybrid')
    extract_noise = (model_type == 'noise_cnn')
    
    # Create datasets
    train_dataset = RealVsAIDataset(
        train_dir, 
        transform=train_transform,
        extract_features=extract_features,
        extract_noise=extract_noise
    )
    
    val_dataset = RealVsAIDataset(
        val_dir,
        transform=val_transform,
        extract_features=extract_features,
        extract_noise=extract_noise
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


class BalancedBatchSampler(torch.utils.data.Sampler):
    """Sampler that ensures balanced batches (equal real and AI images)"""
    
    def __init__(self, dataset: RealVsAIDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Separate indices by class
        self.real_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
        self.ai_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
        
        self.n_batches = min(len(self.real_indices), len(self.ai_indices)) * 2 // batch_size
    
    def __iter__(self):
        # Shuffle indices
        np.random.shuffle(self.real_indices)
        np.random.shuffle(self.ai_indices)
        
        # Create balanced batches
        for i in range(self.n_batches):
            half_batch = self.batch_size // 2
            batch_real = self.real_indices[i*half_batch:(i+1)*half_batch]
            batch_ai = self.ai_indices[i*half_batch:(i+1)*half_batch]
            batch = batch_real + batch_ai
            np.random.shuffle(batch)
            yield from batch
    
    def __len__(self):
        return self.n_batches * self.batch_size
