"""
Quick Start Script for Lightweight Real vs AI Detector
Run this to verify your setup is working
"""

import torch
import sys
from pathlib import Path

def check_cuda():
    """Check CUDA availability"""
    print("🔍 Checking GPU...")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        return True
    else:
        print("   ⚠️  CUDA not available - will use CPU (slower)")
        return False

def check_imports():
    """Check all required imports"""
    print("\n🔍 Checking imports...")
    
    required_modules = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm')
    ]
    
    all_good = True
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NOT INSTALLED")
            all_good = False
    
    return all_good

def check_models():
    """Check model imports"""
    print("\n🔍 Checking custom modules...")
    
    try:
        from src.features import SimpleFeaturesExtractor
        print("   ✅ simple_features.py")
        
        from src.models import LightweightHybridDetector
        print("   ✅ lightweight_model.py")
        
        from src.data import LightweightRealVsAIDataset
        print("   ✅ lightweight_dataset.py")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_model():
    """Test model forward pass"""
    print("\n🔍 Testing model forward pass...")
    
    try:
        from src.models import get_model
        
        # Create dummy data
        img = torch.randn(1, 3, 224, 224)
        stats = torch.randn(1, 6)
        
        # Test hybrid model
        model = get_model('hybrid', pretrained=False)
        
        if torch.cuda.is_available():
            model = model.cuda()
            img = img.cuda()
            stats = stats.cuda()
        
        output = model(img, stats)
        
        print(f"   ✅ Model working")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_data_structure():
    """Check if data directory exists"""
    print("\n🔍 Checking data directory...")
    
    data_dir = Path('data')
    if not data_dir.exists():
        print("   ⚠️  'data' directory not found")
        print("\n   📁 Create this structure:")
        print("   data/")
        print("   ├── train/")
        print("   │   ├── real/")
        print("   │   └── ai/")
        print("   └── val/")
        print("       ├── real/")
        print("       └── ai/")
        return False
    
    # Check subdirectories
    required_dirs = [
        'data/train/real',
        'data/train/ai',
        'data/val/real',
        'data/val/ai'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            count = len(list(Path(dir_path).glob('*.jpg'))) + len(list(Path(dir_path).glob('*.png')))
            print(f"   ✅ {dir_path} ({count} images)")
        else:
            print(f"   ⚠️  {dir_path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def print_next_steps(has_data):
    """Print next steps"""
    print("\n" + "="*60)
    print("📋 NEXT STEPS")
    print("="*60)
    
    if not has_data:
        print("\n1. 📂 Prepare your dataset:")
        print("   - Create data/train/real/ and data/train/ai/")
        print("   - Add 1,000+ images to each folder")
        print("   - Create data/val/real/ and data/val/ai/")
        print("   - Add 200+ validation images")
        print("\n   Suggested datasets:")
        print("   - Kaggle: 'AI vs Real Faces'")
        print("   - Generate AI: Stable Diffusion, Midjourney")
        print("   - Real photos: FFHQ, your camera")
    
    print("\n2. 🏋️ Train your model:")
    print("   python train_lightweight.py --model hybrid --epochs 50")
    
    print("\n3. 📈 Evaluate performance:")
    print("   python evaluate.py --checkpoint checkpoints/hybrid_best.pth --test-dir data/val")
    
    print("\n4. 🔮 Run inference:")
    print("   python inference.py --checkpoint checkpoints/hybrid_best.pth --image test.jpg")
    
    print("\n📖 For detailed instructions, see: LIGHTWEIGHT_GUIDE.md")
    print("="*60)

def main():
    """Run all checks"""
    print("="*60)
    print("🚀 LIGHTWEIGHT REAL VS AI DETECTOR - QUICK START")
    print("="*60)
    
    # Run checks
    cuda_ok = check_cuda()
    imports_ok = check_imports()
    models_ok = check_models()
    
    if not imports_ok or not models_ok:
        print("\n❌ Setup incomplete. Please install missing packages.")
        print("   Run: pip install -r requirements_lightweight.txt")
        sys.exit(1)
    
    # Test model
    model_ok = test_model()
    
    # Check data
    has_data = check_data_structure()
    
    # Summary
    print("\n" + "="*60)
    print("✅ SETUP VERIFICATION COMPLETE")
    print("="*60)
    print(f"   GPU: {'✅' if cuda_ok else '⚠️  CPU only'}")
    print(f"   Packages: {'✅' if imports_ok else '❌'}")
    print(f"   Models: {'✅' if models_ok else '❌'}")
    print(f"   Test: {'✅' if model_ok else '❌'}")
    print(f"   Data: {'✅' if has_data else '⚠️  Not ready'}")
    print("="*60)
    
    # Next steps
    print_next_steps(has_data)
    
    if cuda_ok and imports_ok and models_ok and model_ok:
        print("\n🎉 You're ready to train! Good luck!")
    else:
        print("\n⚠️  Fix the issues above before training.")

if __name__ == "__main__":
    main()
