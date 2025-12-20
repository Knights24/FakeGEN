"""
Verify GPU Setup and Dependencies
=================================
Run this script to verify your environment is correctly configured.
"""

import sys

def check_gpu():
    """Check CUDA and GPU availability."""
    print("="*60)
    print("GPU VERIFICATION")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch Version: {torch.__version__}")
        print(f"✓ CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA Version: {torch.version.cuda}")
            print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
            
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1e9
            print(f"✓ VRAM: {vram_gb:.2f} GB")
            print(f"✓ Compute Capability: {props.major}.{props.minor}")
            
            # Test GPU computation
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.matmul(x, x)
            print(f"✓ GPU Computation Test: PASSED")
        else:
            print("✗ CUDA not available - training will be slow on CPU")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    return True


def check_dependencies():
    """Check all required dependencies."""
    print("\n" + "="*60)
    print("DEPENDENCY CHECK")
    print("="*60)
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'timm': 'PyTorch Image Models',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'sklearn': 'Scikit-learn',
        'albumentations': 'Albumentations',
        'exifread': 'ExifRead',
        'tqdm': 'TQDM',
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            if module == 'PIL':
                import PIL
                version = PIL.__version__
            elif module == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'installed')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_model():
    """Test model initialization."""
    print("\n" + "="*60)
    print("MODEL TEST")
    print("="*60)
    
    try:
        import torch
        from src.models import MultiStreamDeepfakeDetector
        
        # Initialize model
        model = MultiStreamDeepfakeDetector(num_classes=2, pretrained=False)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model initialized successfully")
        print(f"✓ Total Parameters: {total_params:,}")
        print(f"✓ Trainable Parameters: {trainable_params:,}")
        
        # Test forward pass
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        dummy_image = torch.randn(2, 3, 224, 224).to(device)
        dummy_metadata = torch.randn(2, 7).to(device)
        
        with torch.no_grad():
            output = model(dummy_image, dummy_metadata)
        
        print(f"✓ Forward pass successful")
        print(f"✓ Output shape: {output.shape}")
        
        # Get fusion weights
        weights = model.get_fusion_weights()
        print(f"✓ Fusion weights: {weights}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_features():
    """Test feature extractors."""
    print("\n" + "="*60)
    print("FEATURE EXTRACTOR TEST")
    print("="*60)
    
    try:
        import torch
        from src.features import (
            DCTFeatureExtractor,
            SRMExtractor,
            MetadataAnalyzer,
            compute_pixel_correlation
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Test DCT
        dct = DCTFeatureExtractor(block_size=8)
        test_input = torch.randn(2, 3, 224, 224)
        dct_output = dct(test_input)
        print(f"✓ DCT Extractor: Output shape {dct_output.shape}")
        
        # Test SRM
        srm = SRMExtractor()
        srm_output = srm(test_input)
        print(f"✓ SRM Extractor: Output shape {srm_output.shape}")
        
        # Test Metadata
        meta = MetadataAnalyzer()
        print(f"✓ Metadata Analyzer: {meta.NUM_FEATURES} features")
        
        # Test Pixel Correlation
        import numpy as np
        test_image = np.random.rand(224, 224, 3)
        corr = compute_pixel_correlation(test_image)
        print(f"✓ Pixel Correlation: {len(corr)} coefficients")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION - ENVIRONMENT VERIFICATION")
    print("="*60 + "\n")
    
    results = []
    
    results.append(("GPU Setup", check_gpu()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Model", check_model()))
    results.append(("Features", check_features()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All checks passed! Ready for training.")
        print("\nNext steps:")
        print("  1. Prepare your dataset in ./datasets/deepdetect-2025/")
        print("  2. Run: python train.py")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
