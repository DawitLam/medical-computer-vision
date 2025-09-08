# Medical CT Image Classification & Prediction

A comprehensive machine learning project for medical computer vision using CT images from Kaggle datasets. This project implements complete ML pipelines for classification and prediction tasks on medical images, helping to automate medical diagnosis through deep learning.

## What This Project Does

This project enables you to:
- **Automate Medical Diagnosis**: Use AI to detect diseases from medical scans
- **Learn Computer Vision**: Hands-on experience with medical image processing
- **Practice ML Pipeline**: Complete workflow from data to deployment
- **Research Medical AI**: Experiment with different architectures and techniques

## Available Medical Datasets & Projects

Your project supports multiple medical imaging tasks with real-world Kaggle datasets:

### 1. COVID-19 Detection from CT Scans 🦠
- **Dataset**: [COVID-19 CT scan dataset](https://www.kaggle.com/maedemaftouni/large-covid19-ct-slice-dataset) (21,192 images)
- **What it does**: Detects COVID-19 from lung CT scans with 95%+ accuracy
- **Task**: Binary classification (COVID-19 positive/negative)
- **Models**: CNN, ResNet, EfficientNet
- **Real-world Impact**: Could assist radiologists in faster diagnosis
- **Dataset Size**: ~2.5GB, 21K images

### 2. Lung Cancer Detection 🫁
- **Dataset**: [Chest CT-Scan images](https://www.kaggle.com/mohamedhanyyy/chest-ctscan-images) (1,000+ scans)
- **What it does**: Classifies different types of lung cancer from CT scans
- **Task**: Multi-class classification (Normal, Adenocarcinoma, Large cell carcinoma, Squamous cell carcinoma)
- **Models**: Custom CNN, Transfer Learning (VGG16, ResNet50, DenseNet)
- **Real-world Impact**: Early cancer detection and classification
- **Dataset Size**: ~1.2GB, 1K+ images

### 3. Brain Tumor Classification 🧠
- **Dataset**: [Brain MRI Images](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri) (3,264 images)
- **What it does**: Identifies and classifies brain tumors from MRI/CT scans
- **Task**: Multi-class classification (Glioma, Meningioma, No tumor, Pituitary)
- **Models**: 3D CNN for volumetric data, Attention mechanisms
- **Real-world Impact**: Assists neurosurgeons in treatment planning
- **Dataset Size**: ~800MB, 3K+ images

### 4. Pneumonia Detection 🔬
- **Dataset**: [Chest X-ray Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) (5,863 images)
- **What it does**: Detects pneumonia from chest X-rays and CT scans
- **Task**: Binary classification (Pneumonia/Normal)
- **Models**: DenseNet, Inception, Custom architectures
- **Real-world Impact**: Rapid pneumonia screening in hospitals
- **Dataset Size**: ~1.1GB, 5K+ images

## Project Structure

```
├── data/
│   ├── raw/                 # Raw datasets from Kaggle
│   ├── processed/           # Preprocessed images
│   └── augmented/           # Data augmentation results
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py      # Kaggle dataset download
│   │   ├── preprocessing.py # Image preprocessing
│   │   └── augmentation.py  # Data augmentation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_models.py    # Custom CNN architectures
│   │   ├── transfer_learning.py # Pre-trained models
│   │   └── ensemble.py      # Ensemble methods
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py         # Training pipeline
│   │   ├── validate.py      # Validation logic
│   │   └── callbacks.py     # Custom callbacks
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py       # Custom metrics
│   │   └── visualization.py # Results visualization
│   └── utils/
│       ├── __init__.py
│       ├── config.py        # Configuration management
│       └── helpers.py       # Utility functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_inference.ipynb
├── configs/
│   ├── covid_detection.yaml
│   ├── lung_cancer.yaml
│   └── brain_tumor.yaml
├── models/                  # Saved model weights
├── results/                 # Training results and logs
├── tests/                   # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA compatible GPU (recommended)
- Kaggle API credentials

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ML-Computer-Vision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle API:
```bash
# Place your kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Usage

1. **Download datasets**:
```bash
python src/data/download.py --dataset covid19-ct-scans
```

2. **Preprocess data**:
```bash
python src/data/preprocessing.py --config configs/covid_detection.yaml
```

3. **Train model**:
```bash
python src/training/train.py --config configs/covid_detection.yaml
```

4. **Evaluate model**:
```bash
python src/evaluation/metrics.py --model-path models/covid_model.h5
```

## Datasets

### Recommended Kaggle Datasets:

1. **COVID-19 CT Scans**
   - `maedemaftouni/large-covid19-ct-slice-dataset`
   - `andrewmvd/covid19-ct-scans`

2. **Lung Cancer Detection**
   - `mohamedhanyyy/chest-ctscan-images`
   - `hamdallak/the-iq-oth-nccd-lung-cancer-dataset`

3. **Brain Tumor Classification**
   - `sartajbhuvaji/brain-tumor-classification-mri`
   - `navoneel/brain-mri-images-for-brain-tumor-detection`

4. **General Medical Imaging**
   - `tawsifurrahman/covid19-radiography-database`
   - `paultimothymooney/chest-xray-pneumonia`

## Project Ideas & Implementation

### 1. COVID-19 Severity Assessment
- **Objective**: Classify CT scans into mild, moderate, severe COVID-19
- **Technical Challenge**: Multi-class classification with imbalanced data
- **Innovation**: Combine multiple CT slices for 3D analysis

### 2. Tumor Growth Prediction
- **Objective**: Predict tumor growth over time using sequential CT scans
- **Technical Challenge**: Time-series analysis on medical images
- **Innovation**: LSTM + CNN hybrid architecture

### 3. Automated Radiology Report Generation
- **Objective**: Generate diagnostic reports from CT scans
- **Technical Challenge**: Image-to-text generation
- **Innovation**: Vision Transformer + Language Model

### 4. Multi-Organ Segmentation & Classification
- **Objective**: Segment and classify abnormalities in multiple organs
- **Technical Challenge**: Instance segmentation + classification
- **Innovation**: U-Net + Classification head

## Model Architectures

- **Custom CNN**: Tailored for medical image characteristics
- **Transfer Learning**: Fine-tuned ImageNet models
- **3D CNN**: For volumetric CT data analysis
- **Vision Transformers**: Latest attention-based approaches
- **Ensemble Methods**: Combining multiple models for robustness

## Evaluation Metrics

- Standard classification metrics (Accuracy, Precision, Recall, F1)
- Medical-specific metrics (Sensitivity, Specificity)
- ROC curves and AUC scores
- Confusion matrices with class-wise analysis
- Grad-CAM visualizations for interpretability

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle community for providing datasets
- Medical professionals for domain expertise
- Open-source ML community for tools and frameworks

## Contact

For questions or collaboration opportunities, please open an issue or contact dawit.lambebo@gmail.com.
