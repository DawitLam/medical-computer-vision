"""
Environment test script for Medical Computer Vision project.
"""

import sys
import os

def test_environment():
    """Test that all required libraries are installed and working."""
    print("🏥 Medical Computer Vision - Environment Test")
    print("=" * 50)
    
    # Test Python version
    print(f"✓ Python: {sys.version.split()[0]}")
    
    # Test core libraries
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas: {pd.__version__}")
    except ImportError:
        print("✗ Pandas not installed")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow: {tf.__version__}")
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        print(f"✓ GPU Available: {gpu_available}")
        if gpu_available:
            gpus = tf.config.list_physical_devices('GPU')
            print(f"  GPU Devices: {len(gpus)}")
    except ImportError:
        print("✗ TensorFlow not installed")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not installed")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib not installed")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("✗ Scikit-learn not installed")
        return False
    
    # Test project structure
    print("\n📁 Project Structure:")
    required_dirs = ['src', 'data', 'models', 'notebooks', 'configs']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/ directory exists")
        else:
            print(f"✗ {dir_name}/ directory missing")
            return False
    
    # Test project modules
    print("\n🔧 Project Modules:")
    sys.path.append('src')
    
    try:
        from data.download import KaggleDatasetDownloader
        print("✓ Data download module")
    except ImportError as e:
        print(f"⚠ Data download module: {e}")
    
    try:
        from data.preprocessing import MedicalImagePreprocessor
        print("✓ Preprocessing module")
    except ImportError as e:
        print(f"⚠ Preprocessing module: {e}")
    
    try:
        from models.cnn_models import MedicalCNN, create_model
        print("✓ CNN models module")
    except ImportError as e:
        print(f"⚠ CNN models module: {e}")
    
    print("\n🚀 Environment Status: READY!")
    print("You can now:")
    print("  1. Download datasets: python src/data/download.py --list")
    print("  2. Train models: python src/training/train.py --config configs/covid_detection.yaml")
    print("  3. Use Jupyter notebooks: jupyter lab notebooks/")
    print("  4. Run on Google Colab: Upload colab_setup.ipynb")
    
    return True

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)
