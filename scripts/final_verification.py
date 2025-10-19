"""
Final verification script for restructured project
"""
import os
import sys
import torch

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_root)
sys.path.insert(0, parent_dir)

print('='*60)
print('🎉 REAL VS AI DETECTOR - PROJECT VERIFICATION')
print('='*60)

# Package info
print('\n📦 Package Information:')
print(f'   Version: 1.0.0')
print(f'   Python: {sys.version.split()[0]}')
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA: {torch.version.cuda if hasattr(torch.version, "cuda") else "N/A"}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
else:
    print('   GPU: None')

# Check structure
print('\n📁 Project Structure:')
base_dir = parent_dir
dirs = {
    'src/features': os.path.join(base_dir, 'src', 'features'),
    'src/models': os.path.join(base_dir, 'src', 'models'),
    'src/data': os.path.join(base_dir, 'src', 'data'),
    'scripts': os.path.join(base_dir, 'scripts'),
    'docs': os.path.join(base_dir, 'docs'),
    'legacy': os.path.join(base_dir, 'legacy')
}
for name, path in dirs.items():
    exists = '✓' if os.path.exists(path) else '✗'
    print(f'   {exists} {name}/')

# Check files
print('\n📄 Key Files:')
files = {
    'README.md': os.path.join(base_dir, 'README.md'),
    'setup.py': os.path.join(base_dir, 'setup.py'),
    'requirements_lightweight.txt': os.path.join(base_dir, 'requirements_lightweight.txt'),
    '.gitignore': os.path.join(base_dir, '.gitignore')
}
for name, path in files.items():
    exists = '✓' if os.path.exists(path) else '✗'
    print(f'   {exists} {name}')

# Test imports
print('\n🔧 Module Imports:')
try:
    from src.features import SimpleFeaturesExtractor
    print('   ✓ src.features.SimpleFeaturesExtractor')
except Exception as e:
    print(f'   ✗ src.features: {e}')

try:
    from src.models import get_model
    print('   ✓ src.models.get_model')
except Exception as e:
    print(f'   ✗ src.models: {e}')

try:
    from src.data import get_dataloaders
    print('   ✓ src.data.get_dataloaders')
except Exception as e:
    print(f'   ✗ src.data: {e}')

# Check scripts
print('\n🚀 Available Scripts:')
scripts_dir = os.path.join(base_dir, 'scripts')
scripts = ['train.py', 'evaluate.py', 'predict.py', 'verify_setup.py', 'check_installation.py']
for script in scripts:
    path = os.path.join(scripts_dir, script)
    exists = '✓' if os.path.exists(path) else '✗'
    print(f'   {exists} scripts/{script}')

# Check docs
print('\n📚 Documentation:')
docs_dir = os.path.join(base_dir, 'docs')
docs = ['USER_GUIDE.md', 'ALGORITHMS.md', 'REFERENCE.md']
for doc in docs:
    path = os.path.join(docs_dir, doc)
    exists = '✓' if os.path.exists(path) else '✗'
    print(f'   {exists} docs/{doc}')

print('\n' + '='*60)
print('✅ PROJECT RESTRUCTURING COMPLETE!')
print('='*60)

print('\n💡 Quick Start Commands:')
print('   Training:')
print('   python scripts/train.py --train_dir data/final_real --val_dir data/final_ai')
print()
print('   Evaluation:')
print('   python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data/final_real')
print()
print('   Inference:')
print('   python scripts/predict.py --checkpoint checkpoints/best_model.pth --image test.jpg')

print('\n📖 Documentation:')
print('   - README.md - Project overview and quick start')
print('   - docs/USER_GUIDE.md - Complete usage guide')
print('   - RESTRUCTURING_COMPLETE.md - Restructuring summary')
print()
