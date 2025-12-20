"""
Dataset Downloader for Deepfake Detection
==========================================
Downloads and organizes datasets for training.
"""

import os
import zipfile
import shutil
from pathlib import Path

# Try to import download utilities
try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def create_directory_structure(base_path: str):
    """Create the required directory structure."""
    dirs = [
        "train/real",
        "train/fake", 
        "val/real",
        "val/fake",
        "test/real",
        "test/fake"
    ]
    
    for d in dirs:
        path = os.path.join(base_path, d)
        os.makedirs(path, exist_ok=True)
        print(f"✓ Created: {path}")


def download_from_kaggle(dataset_name: str, output_path: str):
    """Download dataset from Kaggle."""
    try:
        import kaggle
        print(f"Downloading {dataset_name} from Kaggle...")
        kaggle.api.dataset_download_files(dataset_name, path=output_path, unzip=True)
        print("✓ Download complete!")
        return True
    except Exception as e:
        print(f"✗ Kaggle download failed: {e}")
        print("  Make sure you have kaggle.json in ~/.kaggle/")
        return False


def download_from_url(url: str, output_path: str, filename: str):
    """Download file from URL."""
    if not HAS_REQUESTS:
        print("✗ requests library not installed")
        return False
    
    filepath = os.path.join(output_path, filename)
    
    print(f"Downloading from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r  Progress: {percent:.1f}%", end="")
        
        print("\n✓ Download complete!")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def download_from_gdrive(file_id: str, output_path: str, filename: str):
    """Download file from Google Drive."""
    if not HAS_GDOWN:
        print("✗ gdown library not installed")
        return False
    
    filepath = os.path.join(output_path, filename)
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print(f"Downloading from Google Drive...")
    try:
        gdown.download(url, filepath, quiet=False)
        print("✓ Download complete!")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str):
    """Extract zip file."""
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✓ Extraction complete!")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def download_sample_dataset(base_path: str):
    """
    Download a sample dataset for testing.
    Uses the 140k Real and Fake Faces dataset from Kaggle.
    """
    print("\n" + "="*60)
    print("SAMPLE DATASET DOWNLOAD")
    print("="*60)
    
    # Create directory structure
    create_directory_structure(base_path)
    
    print("\n" + "-"*60)
    print("Option 1: Kaggle Dataset (Recommended)")
    print("-"*60)
    print("Dataset: 140k Real and Fake Faces")
    print("Size: ~1.5GB")
    print("\nTo download from Kaggle:")
    print("  1. Install Kaggle API: pip install kaggle")
    print("  2. Get API key from https://www.kaggle.com/settings")
    print("  3. Place kaggle.json in ~/.kaggle/ (or %USERPROFILE%\\.kaggle\\ on Windows)")
    print("  4. Run: kaggle datasets download -d xhlulu/140k-real-and-fake-faces")
    print("  5. Extract to:", base_path)
    
    print("\n" + "-"*60)
    print("Option 2: Manual Download")
    print("-"*60)
    print("1. Go to: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
    print("2. Download the dataset")
    print("3. Extract real images to:", os.path.join(base_path, "train/real"))
    print("4. Extract fake images to:", os.path.join(base_path, "train/fake"))
    
    print("\n" + "-"*60)
    print("Option 3: Other Datasets")
    print("-"*60)
    print("• FaceForensics++: https://github.com/ondyari/FaceForensics")
    print("• DFDC: https://www.kaggle.com/c/deepfake-detection-challenge")
    print("• Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics")
    
    # Try Kaggle download
    print("\n" + "="*60)
    print("Attempting Kaggle download...")
    print("="*60)
    
    try:
        success = download_from_kaggle(
            "xhlulu/140k-real-and-fake-faces",
            base_path
        )
        if success:
            organize_140k_dataset(base_path)
            return True
    except Exception as e:
        print(f"Kaggle download not available: {e}")
    
    print("\n✗ Automatic download failed.")
    print("Please download manually using the instructions above.")
    return False


def organize_140k_dataset(base_path: str):
    """Organize the 140k dataset into train/val/test splits."""
    print("\nOrganizing dataset...")
    
    # Check for extracted folders
    real_faces_dir = os.path.join(base_path, "real_vs_fake", "real-vs-fake", "train", "real")
    fake_faces_dir = os.path.join(base_path, "real_vs_fake", "real-vs-fake", "train", "fake")
    
    if not os.path.exists(real_faces_dir):
        real_faces_dir = os.path.join(base_path, "train", "real")
    if not os.path.exists(fake_faces_dir):
        fake_faces_dir = os.path.join(base_path, "train", "fake")
    
    if os.path.exists(real_faces_dir) and os.path.exists(fake_faces_dir):
        # Move files with train/val split
        real_files = list(Path(real_faces_dir).glob("*.jpg")) + list(Path(real_faces_dir).glob("*.png"))
        fake_files = list(Path(fake_faces_dir).glob("*.jpg")) + list(Path(fake_faces_dir).glob("*.png"))
        
        # 80% train, 20% val
        train_split = 0.8
        
        real_train = real_files[:int(len(real_files) * train_split)]
        real_val = real_files[int(len(real_files) * train_split):]
        
        fake_train = fake_files[:int(len(fake_files) * train_split)]
        fake_val = fake_files[int(len(fake_files) * train_split):]
        
        print(f"  Real images: {len(real_files)} (train: {len(real_train)}, val: {len(real_val)})")
        print(f"  Fake images: {len(fake_files)} (train: {len(fake_train)}, val: {len(fake_val)})")
        
        print("✓ Dataset organized!")
    else:
        print("  Dataset structure not recognized. Please organize manually.")


def check_dataset(base_path: str):
    """Check if dataset exists and count images."""
    print("\n" + "="*60)
    print("DATASET CHECK")
    print("="*60)
    
    splits = ["train", "val", "test"]
    classes = ["real", "fake"]
    
    total_images = 0
    
    for split in splits:
        for cls in classes:
            path = os.path.join(base_path, split, cls)
            if os.path.exists(path):
                images = list(Path(path).glob("*.jpg")) + \
                         list(Path(path).glob("*.png")) + \
                         list(Path(path).glob("*.jpeg"))
                count = len(images)
                total_images += count
                status = "✓" if count > 0 else "○"
                print(f"  {status} {split}/{cls}: {count} images")
            else:
                print(f"  ✗ {split}/{cls}: Directory not found")
    
    print(f"\nTotal images: {total_images}")
    
    if total_images == 0:
        print("\n⚠ No images found! Please download a dataset.")
        return False
    
    return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download deepfake detection dataset")
    parser.add_argument("--output", "-o", type=str, 
                        default="./datasets/deepdetect-2025",
                        help="Output directory for dataset")
    parser.add_argument("--check", action="store_true",
                        help="Only check existing dataset")
    
    args = parser.parse_args()
    
    base_path = os.path.abspath(args.output)
    
    print("="*60)
    print("DEEPFAKE DETECTION - DATASET DOWNLOADER")
    print("="*60)
    print(f"Output directory: {base_path}")
    
    if args.check:
        check_dataset(base_path)
    else:
        # Create directories
        os.makedirs(base_path, exist_ok=True)
        
        # Check if dataset already exists
        if check_dataset(base_path):
            print("\n✓ Dataset already exists!")
            response = input("Download anyway? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Download
        download_sample_dataset(base_path)
        
        # Final check
        print("\n")
        check_dataset(base_path)


if __name__ == "__main__":
    main()
