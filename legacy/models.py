"""
Model Architectures for Real vs AI Image Detection

Implements:
1. CNN-based Classifier (ResNet-style)
2. Hybrid CNN + Statistical Features Model
3. Noise Residual CNN (NR-CNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple


class ResidualBlock(nn.Module):
    """Residual block with two conv layers"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNNClassifier(nn.Module):
    """
    CNN-based Binary Classifier
    
    Architecture:
        Input (224x224x3)
        → Conv2D → ReLU → BatchNorm → MaxPool
        → Residual Blocks (ResNet-style)
        → Global Average Pooling
        → Dense(512) → Dropout → Dense(1)
        
    Loss: Binary Cross Entropy
    L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
    """
    
    def __init__(self, num_residual_blocks: int = 4, dropout: float = 0.5):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_residual_blocks, stride=1)
        self.layer2 = self._make_layer(64, 128, num_residual_blocks, stride=2)
        self.layer3 = self._make_layer(128, 256, num_residual_blocks, stride=2)
        self.layer4 = self._make_layer(256, 512, num_residual_blocks, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head
        self.fc1 = nn.Linear(512, 512)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 1)
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class HybridModel(nn.Module):
    """
    Hybrid CNN + Statistical Features Model
    
    Architecture:
        CNN Branch: Extract deep features (f_cnn)
        Statistical Branch: Noise, correlation, DCT, PRNU features (f_stat)
        Metadata Branch: EXIF features (f_meta)
        
        Concatenate: h = [f_cnn, f_stat, f_meta]
        Final: ŷ = σ(W^T h + b)
    """
    
    def __init__(self, stat_feature_dim: int = 14, meta_feature_dim: int = 7,
                 dropout: float = 0.5, pretrained_cnn: bool = True):
        super().__init__()
        
        # CNN branch (using EfficientNet-B0 for efficiency)
        if pretrained_cnn:
            self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            # Remove final classification layer
            self.cnn.classifier = nn.Identity()
            cnn_out_features = 1280
        else:
            # Use our custom CNN
            self.cnn = CNNClassifier(num_residual_blocks=2)
            self.cnn.fc2 = nn.Identity()  # Remove final layer
            cnn_out_features = 512
        
        # Statistical features branch
        self.stat_branch = nn.Sequential(
            nn.Linear(stat_feature_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Metadata features branch
        self.meta_branch = nn.Sequential(
            nn.Linear(meta_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        
        # Fusion layer
        fusion_dim = cnn_out_features + 64 + 16
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, image, stat_features, meta_features):
        """
        Args:
            image: (B, 3, H, W) RGB image tensor
            stat_features: (B, stat_feature_dim) statistical features
            meta_features: (B, meta_feature_dim) metadata features
        
        Returns:
            logits: (B, 1) binary classification logits
        """
        # CNN features
        f_cnn = self.cnn(image)
        if len(f_cnn.shape) > 2:
            f_cnn = torch.flatten(f_cnn, 1)
        
        # Statistical features
        f_stat = self.stat_branch(stat_features)
        
        # Metadata features
        f_meta = self.meta_branch(meta_features)
        
        # Concatenate all features
        h = torch.cat([f_cnn, f_stat, f_meta], dim=1)
        
        # Final classifier
        logits = self.fusion(h)
        
        return logits


class NoiseResidualCNN(nn.Module):
    """
    Noise Residual CNN (NR-CNN)
    
    Trains directly on noise residual maps extracted from images.
    Useful for detecting GAN fingerprints and subtle artifacts.
    """
    
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        
        # Input: Noise residual map (single channel)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, noise_map):
        """
        Args:
            noise_map: (B, 1, H, W) noise residual map
        
        Returns:
            logits: (B, 1) binary classification logits
        """
        features = self.features(noise_map)
        logits = self.classifier(features)
        return logits


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based classifier for baseline comparison
    """
    
    def __init__(self, model_name: str = 'efficientnet_b0', 
                 pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        
        # Load pretrained EfficientNet
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            num_features = 1280
        elif model_name == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(
                weights=models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            )
            num_features = 1280
        elif model_name == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(
                weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            )
            num_features = 1408
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.backbone(x)


def get_model(model_type: str = 'cnn', **kwargs) -> nn.Module:
    """
    Factory function to get model by type
    
    Args:
        model_type: 'cnn', 'hybrid', 'noise_cnn', 'efficientnet'
        **kwargs: Model-specific arguments
    
    Returns:
        PyTorch model
    """
    if model_type == 'cnn':
        return CNNClassifier(**kwargs)
    elif model_type == 'hybrid':
        return HybridModel(**kwargs)
    elif model_type == 'noise_cnn':
        return NoiseResidualCNN(**kwargs)
    elif model_type == 'efficientnet':
        return EfficientNetClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("Testing CNN Classifier...")
    model = CNNClassifier()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"CNN output shape: {out.shape}")
    
    print("\nTesting Hybrid Model...")
    model = HybridModel()
    stat_feat = torch.randn(2, 14)
    meta_feat = torch.randn(2, 7)
    out = model(x, stat_feat, meta_feat)
    print(f"Hybrid output shape: {out.shape}")
    
    print("\nTesting Noise Residual CNN...")
    model = NoiseResidualCNN()
    noise = torch.randn(2, 1, 224, 224)
    out = model(noise)
    print(f"Noise CNN output shape: {out.shape}")
    
    print("\nAll models initialized successfully!")
