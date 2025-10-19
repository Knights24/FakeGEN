"""
Lightweight Hybrid Detector: EfficientNet-B0 + Statistical Features
Optimized for RTX 4060 Laptop GPU with memory-efficient design
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class LightweightHybridDetector(nn.Module):
    """
    Hybrid model combining:
    - EfficientNet-B0 backbone (pretrained on ImageNet)
    - Simple statistical features (6 values)
    - Small MLP fusion head
    
    Total params: ~5M (lightweight for deployment)
    """
    
    def __init__(self, 
                 pretrained: bool = True,
                 num_stat_features: int = 6,
                 dropout: float = 0.3,
                 freeze_backbone: bool = False):
        """
        Args:
            pretrained: Use ImageNet pretrained weights
            num_stat_features: Number of statistical features (default: 6)
            dropout: Dropout rate for regularization
            freeze_backbone: Freeze CNN backbone for faster training
        """
        super().__init__()
        
        # EfficientNet-B0 backbone
        if pretrained:
            self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.cnn = models.efficientnet_b0(weights=None)
        
        # Remove classification head (keep feature extractor)
        self.cnn.classifier = nn.Identity()
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        # Fusion head: CNN features (1280) + Statistical features (6)
        self.fusion_head = nn.Sequential(
            nn.Linear(1280 + num_stat_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img: torch.Tensor, stat_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            img: Image tensor [B, 3, 224, 224]
            stat_features: Statistical features [B, 6]
        
        Returns:
            Predictions [B, 1] in range [0, 1]
        """
        # Extract CNN features
        cnn_features = self.cnn(img)  # [B, 1280]
        
        # Concatenate with statistical features
        combined = torch.cat([cnn_features, stat_features], dim=1)  # [B, 1286]
        
        # Fusion head
        output = self.fusion_head(combined)  # [B, 1]
        
        return output
    
    def get_num_params(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleEfficientNetDetector(nn.Module):
    """
    Simpler version: Just EfficientNet-B0 (no statistical features)
    Useful for baseline comparison
    """
    
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        
        if pretrained:
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.backbone(img)


class TinyDetector(nn.Module):
    """
    Ultra-lightweight model for edge deployment
    Custom CNN with minimal parameters (~500K)
    """
    
    def __init__(self, num_stat_features: int = 6):
        super().__init__()
        
        # Tiny CNN backbone
        self.cnn = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Fusion head
        self.fc = nn.Sequential(
            nn.Linear(256 + num_stat_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img: torch.Tensor, stat_features: torch.Tensor) -> torch.Tensor:
        x = self.cnn(img)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, stat_features], dim=1)
        return self.fc(x)


def get_model(model_type: str = 'hybrid', **kwargs) -> nn.Module:
    """
    Factory function to get model
    
    Args:
        model_type: 'hybrid' (EfficientNet + stats), 
                   'efficientnet' (EfficientNet only),
                   'tiny' (lightweight CNN)
        **kwargs: Model-specific arguments
    
    Returns:
        PyTorch model
    """
    if model_type == 'hybrid':
        return LightweightHybridDetector(**kwargs)
    elif model_type == 'efficientnet':
        return SimpleEfficientNetDetector(**kwargs)
    elif model_type == 'tiny':
        return TinyDetector(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("Testing Lightweight Models...")
    
    # Test hybrid model
    print("\n1. Hybrid Detector (EfficientNet-B0 + Stats):")
    model = LightweightHybridDetector(pretrained=False)
    img = torch.randn(2, 3, 224, 224)
    stats = torch.randn(2, 6)
    output = model(img, stats)
    print(f"   Input: img {img.shape}, stats {stats.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Total params: {model.get_num_params():,}")
    print(f"   Trainable params: {model.get_trainable_params():,}")
    
    # Test simple EfficientNet
    print("\n2. Simple EfficientNet Detector:")
    model = SimpleEfficientNetDetector(pretrained=False)
    output = model(img)
    print(f"   Input: {img.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test tiny model
    print("\n3. Tiny Detector (Edge deployment):")
    model = TinyDetector()
    output = model(img, stats)
    print(f"   Input: img {img.shape}, stats {stats.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        print("\n4. Testing on GPU:")
        device = torch.device('cuda')
        model = LightweightHybridDetector(pretrained=False).to(device)
        img_gpu = img.to(device)
        stats_gpu = stats.to(device)
        output = model(img_gpu, stats_gpu)
        print(f"   ✓ GPU forward pass successful")
        print(f"   Output device: {output.device}")
    
    print("\n✅ All models working!")
