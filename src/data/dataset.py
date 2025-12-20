import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Callable, Dict, List, Optional, Tuple

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    import torchvision.transforms as T

from ..features.metadata_analyzer import MetadataAnalyzer


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', transform: Optional[Callable] = None,
                 include_metadata: bool = True, img_size: int = 224):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.include_metadata = include_metadata
        self.img_size = img_size
        
        if include_metadata:
            self.metadata_analyzer = MetadataAnalyzer()
        else:
            self.metadata_analyzer = None
        
        self.samples: List[Tuple[str, int]] = []
        self._load_samples()
        self._metadata_cache: Dict[str, torch.Tensor] = {}
    
    def _load_samples(self):
        class_mapping = {'real': 0, 'fake': 1}
        split_dir = self.split
        if self.split == 'val':
            if not os.path.exists(os.path.join(self.root_dir, 'val')) and \
               os.path.exists(os.path.join(self.root_dir, 'valid')):
                split_dir = 'valid'
        
        for class_name, label in class_mapping.items():
            class_dir = os.path.join(self.root_dir, split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
            
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            for img_name in os.listdir(class_dir):
                ext = os.path.splitext(img_name)[1].lower()
                if ext in valid_extensions:
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, label))
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {os.path.join(self.root_dir, self.split)}")
        
        print(f"Loaded {len(self.samples)} samples from {self.split} split")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        if self.include_metadata and self.metadata_analyzer is not None:
            if img_path in self._metadata_cache:
                metadata_tensor = self._metadata_cache[img_path]
            else:
                metadata = self.metadata_analyzer.extract_features(img_path)
                metadata_tensor = torch.tensor([
                    metadata['has_exif'], metadata['camera_make'], metadata['camera_model'],
                    metadata['software'], metadata['gps_info'], metadata['color_space'], 
                    metadata['compression']], dtype=torch.float32)
                self._metadata_cache[img_path] = metadata_tensor
        else:
            metadata_tensor = torch.zeros(7, dtype=torch.float32)
        
        if self.transform is not None:
            if ALBUMENTATIONS_AVAILABLE:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                image = Image.fromarray(image)
                image = self.transform(image)
        else:
            image = Image.fromarray(image)
            image = image.resize((self.img_size, self.img_size))
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        return {'image': image, 'metadata': metadata_tensor, 
                'label': torch.tensor(label, dtype=torch.long), 'path': img_path}
    
    def get_class_distribution(self) -> Dict[str, int]:
        distribution = {'real': 0, 'fake': 0}
        for _, label in self.samples:
            if label == 0:
                distribution['real'] += 1
            else:
                distribution['fake'] += 1
        return distribution


def get_transforms(split: str = 'train', img_size: int = 224) -> Optional[Callable]:
    if ALBUMENTATIONS_AVAILABLE:
        if split == 'train':
            return A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.GaussianBlur(blur_limit=(3, 7), p=0.1),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    else:
        if split == 'train':
            return T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])


def create_dataloaders(root_dir: str, batch_size: int = 16, num_workers: int = 4, img_size: int = 224,
                       include_metadata: bool = True, pin_memory: bool = True, persistent_workers: bool = False,
                       prefetch_factor: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_dataset = DeepfakeDataset(
        root_dir=root_dir,
        split='train',
        transform=get_transforms('train', img_size),
        include_metadata=include_metadata,
        img_size=img_size
    )
    
    val_dataset = DeepfakeDataset(
        root_dir=root_dir,
        split='val',
        transform=get_transforms('val', img_size),
        include_metadata=include_metadata,
        img_size=img_size
    )
    
    test_dataset = DeepfakeDataset(
        root_dir=root_dir,
        split='test',
        transform=get_transforms('test', img_size),
        include_metadata=include_metadata,
        img_size=img_size
    )
    
    
    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs['persistent_workers'] = persistent_workers
        worker_kwargs['prefetch_factor'] = prefetch_factor
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **worker_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **worker_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **worker_kwargs
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    images = torch.stack([item['image'] for item in batch])
    metadata = torch.stack([item['metadata'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    paths = [item['path'] for item in batch]
    return {'image': images, 'metadata': metadata, 'label': labels, 'path': paths}