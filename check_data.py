"""
Quick data check script for medical computer vision project.
Checks if data exists and provides recommendations.
"""

import os
from pathlib import Path
import sys

def check_data_status():
    """Check current data status and provide recommendations."""
    
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    
    print("🏥 MEDICAL COMPUTER VISION - DATA STATUS CHECK")
    print("=" * 55)
    
    # Check if data directory exists
    if not data_dir.exists():
        print("❌ Data directory not found!")
        print("📁 Creating data directory...")
        data_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Data directory created at:", data_dir)
    
    # Check for existing data
    data_found = False
    dataset_info = []
    
    # Common medical dataset patterns
    dataset_patterns = [
        "brain*tumor*",
        "covid*",
        "lung*cancer*",
        "pneumonia*",
        "chest*xray*",
        "ct*scan*",
        "*dicom*"
    ]
    
    print(f"\n🔍 Scanning for medical datasets in: {data_dir}")
    
    # Check subdirectories
    if data_dir.exists():
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            print(f"   Found {len(subdirs)} subdirectories:")
            for subdir in subdirs:
                # Count image files
                image_files = []
                for ext in ['.png', '.jpg', '.jpeg', '.dcm', '.dicom', '.tif']:
                    image_files.extend(list(subdir.glob(f'**/*{ext}')))
                    image_files.extend(list(subdir.glob(f'**/*{ext.upper()}')))
                
                if image_files:
                    data_found = True
                    dataset_info.append({
                        'name': subdir.name,
                        'path': str(subdir),
                        'images': len(image_files),
                        'size_mb': sum(f.stat().st_size for f in image_files) / (1024*1024)
                    })
                    print(f"   📂 {subdir.name}: {len(image_files)} images ({sum(f.stat().st_size for f in image_files) / (1024*1024):.1f} MB)")
                else:
                    print(f"   📂 {subdir.name}: No image files found")
        else:
            print("   No subdirectories found")
    
    # Data status summary
    print(f"\n📊 DATA STATUS SUMMARY:")
    if data_found:
        total_images = sum(info['images'] for info in dataset_info)
        total_size = sum(info['size_mb'] for info in dataset_info)
        
        print(f"   ✅ Medical image data found!")
        print(f"   📈 Total datasets: {len(dataset_info)}")
        print(f"   🖼️  Total images: {total_images}")
        print(f"   💾 Total size: {total_size:.1f} MB")
        
        print(f"\n🧹 RECOMMENDED NEXT STEPS:")
        print(f"   1. Run data quality analysis:")
        print(f"      python src/data/data_cleaner.py --input data --analyze-only")
        print(f"   2. If issues found, clean the data:")
        print(f"      python src/data/data_cleaner.py --input data --output data/cleaned")
        print(f"   3. Start training with cleaned data")
        
        # Check for common data quality issues
        print(f"\n⚠️  POTENTIAL ISSUES TO CHECK:")
        print(f"   • Image size consistency")
        print(f"   • File corruption")
        print(f"   • Class distribution balance")
        print(f"   • Duplicate images")
        print(f"   • Missing labels/folder structure")
        
    else:
        print(f"   ❌ No medical image data found!")
        print(f"\n📥 RECOMMENDED ACTIONS:")
        print(f"   1. Download medical datasets using Kaggle API:")
        print(f"      python src/data/download.py")
        print(f"   2. Or manually place datasets in the data/ directory")
        print(f"   3. Supported formats: PNG, JPG, DICOM (.dcm)")
        
        # Suggest popular medical datasets
        print(f"\n🏥 SUGGESTED MEDICAL DATASETS:")
        print(f"   • Brain Tumor MRI Dataset")
        print(f"   • COVID-19 Chest X-ray Dataset")
        print(f"   • Lung Cancer CT Scan Dataset")
        print(f"   • NIH Chest X-ray Dataset")
        print(f"   • LIDC-IDRI Lung CT Dataset")
        
        print(f"\n📖 TO GET STARTED:")
        print(f"   1. Set up Kaggle API: pip install kaggle")
        print(f"   2. Configure API key: ~/.kaggle/kaggle.json")
        print(f"   3. Download data: python src/data/download.py")
    
    # Check Python environment
    print(f"\n🐍 ENVIRONMENT CHECK:")
    try:
        import cv2
        import numpy as np
        import pandas as pd
        print(f"   ✅ OpenCV: {cv2.__version__}")
        print(f"   ✅ NumPy: {np.__version__}")
        print(f"   ✅ Pandas: {pd.__version__}")
        
        # Check for optional packages
        try:
            import pydicom
            print(f"   ✅ PyDICOM: {pydicom.__version__} (for DICOM files)")
        except ImportError:
            print(f"   ⚠️  PyDICOM not installed (pip install pydicom for DICOM support)")
            
    except ImportError as e:
        print(f"   ❌ Missing required package: {e}")
        print(f"   📦 Run: pip install -r requirements.txt")
    
    return data_found, dataset_info

if __name__ == "__main__":
    data_found, dataset_info = check_data_status()
    
    if data_found:
        print(f"\n🚀 READY FOR DATA ANALYSIS!")
        print(f"Run data quality check: python src/data/data_cleaner.py --input data --analyze-only")
    else:
        print(f"\n📥 NEXT: Download medical datasets")
        print(f"Run: python src/data/download.py")
