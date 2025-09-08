"""System capability check for medical computer vision."""

import psutil
import platform
import tensorflow as tf

print("💻 LAPTOP CAPABILITY ASSESSMENT")
print("=" * 50)

# Basic system info
print(f"🖥️  Operating System: {platform.system()} {platform.release()}")
print(f"🧮 CPU: {platform.processor()}")
print(f"⚡ CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")

# Memory info
memory = psutil.virtual_memory()
print(f"💾 RAM Total: {memory.total / (1024**3):.1f} GB")
print(f"💾 RAM Available: {memory.available / (1024**3):.1f} GB")
print(f"💾 RAM Usage: {memory.percent}%")

# Disk space
disk = psutil.disk_usage('.')
print(f"💿 Disk Space Free: {disk.free / (1024**3):.1f} GB")

# GPU check
print(f"\n🎮 GPU STATUS:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Available: {len(gpus)} GPU(s) detected")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")
else:
    print("❌ No GPU detected (CPU-only training)")

print(f"\n📊 CAPABILITY ASSESSMENT:")

# Memory assessment
if memory.total / (1024**3) >= 16:
    memory_rating = "EXCELLENT"
elif memory.total / (1024**3) >= 8:
    memory_rating = "GOOD"
elif memory.total / (1024**3) >= 4:
    memory_rating = "FAIR"
else:
    memory_rating = "LIMITED"

print(f"Memory Rating: {memory_rating}")

# CPU assessment
cpu_cores = psutil.cpu_count(logical=True)
if cpu_cores >= 8:
    cpu_rating = "EXCELLENT"
elif cpu_cores >= 4:
    cpu_rating = "GOOD"
else:
    cpu_rating = "FAIR"

print(f"CPU Rating: {cpu_rating}")

# Overall assessment
print(f"\n🎯 MEDICAL CV TRAINING CAPABILITY:")

if len(gpus) > 0:
    print("✅ EXCELLENT - GPU available for fast training")
    print("   → Training time: 1-3 hours per model")
    print("   → Recommended: Train locally")
elif memory.total / (1024**3) >= 8 and cpu_cores >= 4:
    print("✅ GOOD - CPU training feasible")
    print("   → Training time: 4-12 hours per model")
    print("   → Recommended: Mix of local + cloud training")
elif memory.total / (1024**3) >= 4:
    print("⚠️  FAIR - Limited but workable")
    print("   → Training time: 8-24 hours per model")
    print("   → Recommended: Use Google Colab for training")
else:
    print("❌ LIMITED - Cloud training recommended")
    print("   → Recommended: Use Google Colab or Kaggle")

print(f"\n📋 DATASET SIZE COMPATIBILITY:")
datasets = {
    "COVID-19 CT (21K images)": 2.5,
    "Lung Cancer (1K images)": 1.2,
    "Brain Tumor (3K images)": 0.8,
    "Pneumonia (5K images)": 1.1
}

for dataset, size_gb in datasets.items():
    if disk.free / (1024**3) > size_gb * 3:  # 3x space for processing
        print(f"✅ {dataset}: {size_gb}GB - Can handle locally")
    else:
        print(f"⚠️  {dataset}: {size_gb}GB - Use cloud storage")

print(f"\n🚀 RECOMMENDATIONS:")
if len(gpus) > 0:
    print("1. ✅ Train locally - you have GPU power!")
    print("2. Use Google Colab for experiments")
    print("3. Your laptop is well-suited for this project")
elif memory.total / (1024**3) >= 8:
    print("1. Start with smaller datasets locally")
    print("2. Use Google Colab for larger models")
    print("3. Your laptop can handle development well")
else:
    print("1. ✅ Use Google Colab for training (FREE GPU!)")
    print("2. Use your laptop for code development")
    print("3. Your laptop is perfect for learning and coding")

print("\n💡 Next steps:")
print("   → Test with small dataset first")
print("   → Monitor memory usage during training")
print("   → Use batch size optimization")
