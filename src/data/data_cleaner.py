"""
Medical Image Data Quality Assessment and Cleaning Pipeline

This module analyzes and cleans medical image datasets for optimal ML training.
Handles common issues in medical imaging data.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import pydicom
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalDataCleaner:
    """Comprehensive data cleaning for medical images."""
    
    def __init__(self, data_dir: str):
        """
        Initialize data cleaner.
        
        Args:
            data_dir: Directory containing medical images
        """
        self.data_dir = Path(data_dir)
        self.cleaning_report = {
            'total_files': 0,
            'valid_images': 0,
            'corrupted_files': [],
            'duplicate_images': [],
            'size_issues': [],
            'format_issues': [],
            'empty_files': [],
            'class_distribution': {},
            'image_stats': {}
        }
    
    def analyze_dataset_quality(self) -> Dict:
        """
        Comprehensive analysis of dataset quality issues.
        
        Returns:
            Dictionary with quality assessment results
        """
        logger.info("🔍 Starting medical image dataset quality analysis...")
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.dcm', '.dicom', '.tif', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.data_dir.glob(f'**/*{ext}'))
            image_files.extend(self.data_dir.glob(f'**/*{ext.upper()}'))
        
        self.cleaning_report['total_files'] = len(image_files)
        logger.info(f"Found {len(image_files)} image files")
        
        if len(image_files) == 0:
            logger.warning("No image files found! Check dataset path.")
            return self.cleaning_report
        
        # Analyze each image
        valid_images = []
        image_sizes = []
        image_channels = []
        file_sizes = []
        
        for img_path in image_files[:100]:  # Sample first 100 for speed
            try:
                # Check file size
                file_size = img_path.stat().st_size
                file_sizes.append(file_size)
                
                if file_size == 0:
                    self.cleaning_report['empty_files'].append(str(img_path))
                    continue
                
                # Try to load image
                if img_path.suffix.lower() in ['.dcm', '.dicom']:
                    # DICOM file
                    try:
                        ds = pydicom.dcmread(str(img_path))
                        img = ds.pixel_array
                        if len(img.shape) == 2:
                            img_channels.append(1)
                        else:
                            img_channels.append(img.shape[2])
                        image_sizes.append(img.shape[:2])
                    except Exception as e:
                        self.cleaning_report['corrupted_files'].append({
                            'file': str(img_path),
                            'error': str(e)
                        })
                        continue
                else:
                    # Regular image file
                    img = cv2.imread(str(img_path))
                    if img is None:
                        self.cleaning_report['corrupted_files'].append({
                            'file': str(img_path),
                            'error': 'Could not load with OpenCV'
                        })
                        continue
                    
                    if len(img.shape) == 3:
                        img_channels.append(img.shape[2])
                        image_sizes.append((img.shape[0], img.shape[1]))
                    else:
                        img_channels.append(1)
                        image_sizes.append(img.shape)
                
                valid_images.append(img_path)
                
            except Exception as e:
                self.cleaning_report['corrupted_files'].append({
                    'file': str(img_path),
                    'error': str(e)
                })
        
        self.cleaning_report['valid_images'] = len(valid_images)
        
        # Analyze image statistics
        if image_sizes:
            sizes_array = np.array(image_sizes)
            self.cleaning_report['image_stats'] = {
                'mean_height': float(np.mean(sizes_array[:, 0])),
                'mean_width': float(np.mean(sizes_array[:, 1])),
                'std_height': float(np.std(sizes_array[:, 0])),
                'std_width': float(np.std(sizes_array[:, 1])),
                'min_height': int(np.min(sizes_array[:, 0])),
                'max_height': int(np.max(sizes_array[:, 0])),
                'min_width': int(np.min(sizes_array[:, 1])),
                'max_width': int(np.max(sizes_array[:, 1])),
                'unique_sizes': len(np.unique(sizes_array, axis=0)),
                'channels': list(set(img_channels)),
                'mean_file_size_mb': float(np.mean(file_sizes) / (1024*1024)),
                'std_file_size_mb': float(np.std(file_sizes) / (1024*1024))
            }
        
        # Analyze class distribution
        self._analyze_class_distribution(valid_images)
        
        # Check for duplicate images
        self._check_for_duplicates(valid_images[:50])  # Sample for speed
        
        # Generate quality report
        self._generate_quality_report()
        
        return self.cleaning_report
    
    def _analyze_class_distribution(self, image_files: List[Path]):
        """Analyze class distribution in the dataset."""
        class_counts = {}
        
        for img_path in image_files:
            # Try to extract class from folder structure
            parent_folders = img_path.parts
            
            # Common medical dataset structures
            possible_classes = []
            for folder in parent_folders:
                folder_lower = folder.lower()
                if any(term in folder_lower for term in [
                    'covid', 'normal', 'pneumonia', 'tumor', 'cancer',
                    'benign', 'malignant', 'positive', 'negative',
                    'glioma', 'meningioma', 'pituitary', 'notumor'
                ]):
                    possible_classes.append(folder)
            
            if possible_classes:
                class_name = possible_classes[-1]  # Use the most specific
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        self.cleaning_report['class_distribution'] = class_counts
    
    def _check_for_duplicates(self, image_files: List[Path]):
        """Check for duplicate images using file hashing."""
        import hashlib
        
        file_hashes = {}
        duplicates = []
        
        for img_path in image_files:
            try:
                # Calculate file hash
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                if file_hash in file_hashes:
                    duplicates.append({
                        'original': str(file_hashes[file_hash]),
                        'duplicate': str(img_path)
                    })
                else:
                    file_hashes[file_hash] = img_path
                    
            except Exception as e:
                logger.warning(f"Could not hash {img_path}: {e}")
        
        self.cleaning_report['duplicate_images'] = duplicates
    
    def _generate_quality_report(self):
        """Generate a comprehensive quality report."""
        report = self.cleaning_report
        
        print("\n🏥 MEDICAL IMAGE DATASET QUALITY REPORT")
        print("=" * 60)
        
        # Overall statistics
        print(f"📊 DATASET OVERVIEW:")
        print(f"   Total files found: {report['total_files']}")
        print(f"   Valid images: {report['valid_images']}")
        print(f"   Success rate: {(report['valid_images']/max(report['total_files'],1)*100):.1f}%")
        
        # Issues found
        print(f"\n⚠️  ISSUES DETECTED:")
        print(f"   Corrupted files: {len(report['corrupted_files'])}")
        print(f"   Empty files: {len(report['empty_files'])}")
        print(f"   Duplicate images: {len(report['duplicate_images'])}")
        
        # Image statistics
        if report['image_stats']:
            stats = report['image_stats']
            print(f"\n📏 IMAGE DIMENSIONS:")
            print(f"   Average size: {stats['mean_height']:.0f} x {stats['mean_width']:.0f}")
            print(f"   Size range: {stats['min_height']}x{stats['min_width']} to {stats['max_height']}x{stats['max_width']}")
            print(f"   Unique sizes: {stats['unique_sizes']}")
            print(f"   Channels: {stats['channels']}")
            print(f"   Average file size: {stats['mean_file_size_mb']:.2f} MB")
            
            # Size consistency check
            size_variance = (stats['std_height'] + stats['std_width']) / 2
            if size_variance > 100:
                print(f"   ⚠️ High size variance detected: {size_variance:.1f}")
                print(f"   → Recommendation: Standardize image sizes")
        
        # Class distribution
        if report['class_distribution']:
            print(f"\n🏷️  CLASS DISTRIBUTION:")
            total_labeled = sum(report['class_distribution'].values())
            for class_name, count in report['class_distribution'].items():
                percentage = (count / total_labeled) * 100
                print(f"   {class_name}: {count} images ({percentage:.1f}%)")
            
            # Check for class imbalance
            counts = list(report['class_distribution'].values())
            if len(counts) > 1:
                imbalance_ratio = max(counts) / min(counts)
                if imbalance_ratio > 5:
                    print(f"   ⚠️ Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
                    print(f"   → Recommendation: Use class weighting or data augmentation")
        
        # Recommendations
        print(f"\n🚀 CLEANING RECOMMENDATIONS:")
        
        if len(report['corrupted_files']) > 0:
            print(f"   1. Remove {len(report['corrupted_files'])} corrupted files")
        
        if len(report['duplicate_images']) > 0:
            print(f"   2. Remove {len(report['duplicate_images'])} duplicate images")
        
        if report['image_stats'] and report['image_stats']['unique_sizes'] > 5:
            print(f"   3. Standardize image sizes (current: {report['image_stats']['unique_sizes']} unique sizes)")
        
        if report['image_stats'] and len(report['image_stats']['channels']) > 1:
            print(f"   4. Standardize color channels: {report['image_stats']['channels']}")
        
        print(f"   5. Apply medical-specific preprocessing (HU windowing, normalization)")
        print(f"   6. Quality control: manual review of samples")
    
    def clean_dataset(self, output_dir: str, target_size: Tuple[int, int] = (224, 224)) -> str:
        """
        Clean the dataset and save cleaned images.
        
        Args:
            output_dir: Directory to save cleaned images
            target_size: Target image size (height, width)
            
        Returns:
            Path to cleaned dataset
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🧹 Starting dataset cleaning...")
        logger.info(f"Target size: {target_size}")
        logger.info(f"Output directory: {output_path}")
        
        # Get all valid images
        image_extensions = ['.png', '.jpg', '.jpeg', '.dcm', '.dicom']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.data_dir.glob(f'**/*{ext}'))
        
        cleaned_count = 0
        skipped_count = 0
        
        for img_path in image_files:
            try:
                # Skip if in corrupted list
                if any(str(img_path) in item['file'] if isinstance(item, dict) else str(img_path) in item 
                       for item in self.cleaning_report['corrupted_files']):
                    skipped_count += 1
                    continue
                
                # Load and process image
                if img_path.suffix.lower() in ['.dcm', '.dicom']:
                    # DICOM processing
                    ds = pydicom.dcmread(str(img_path))
                    img = ds.pixel_array.astype(np.float32)
                    
                    # Apply HU windowing if available
                    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                        img = img * ds.RescaleSlope + ds.RescaleIntercept
                    
                    # Convert to 8-bit
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                    
                    # Convert to RGB if grayscale
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        
                else:
                    # Regular image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        skipped_count += 1
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to target size
                img_resized = cv2.resize(img, (target_size[1], target_size[0]))
                
                # Create output path maintaining directory structure
                rel_path = img_path.relative_to(self.data_dir)
                output_file = output_path / rel_path.with_suffix('.png')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save cleaned image
                img_pil = Image.fromarray(img_resized)
                img_pil.save(output_file)
                
                cleaned_count += 1
                
                if cleaned_count % 100 == 0:
                    logger.info(f"Processed {cleaned_count} images...")
                    
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                skipped_count += 1
        
        logger.info(f"✅ Cleaning completed!")
        logger.info(f"   Cleaned images: {cleaned_count}")
        logger.info(f"   Skipped images: {skipped_count}")
        logger.info(f"   Output directory: {output_path}")
        
        return str(output_path)
    
    def create_data_summary(self, output_file: str = "data_quality_report.csv"):
        """Create a CSV summary of data quality issues."""
        summary_data = []
        
        # Add corrupted files
        for item in self.cleaning_report['corrupted_files']:
            if isinstance(item, dict):
                summary_data.append({
                    'file_path': item['file'],
                    'issue_type': 'corrupted',
                    'description': item['error']
                })
        
        # Add empty files
        for file_path in self.cleaning_report['empty_files']:
            summary_data.append({
                'file_path': file_path,
                'issue_type': 'empty',
                'description': 'File size is 0 bytes'
            })
        
        # Add duplicates
        for dup in self.cleaning_report['duplicate_images']:
            summary_data.append({
                'file_path': dup['duplicate'],
                'issue_type': 'duplicate',
                'description': f"Duplicate of {dup['original']}"
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(output_file, index=False)
            logger.info(f"📄 Data quality summary saved to {output_file}")
        else:
            logger.info("✅ No data quality issues found!")


def main():
    """Main function for data cleaning."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean medical image dataset')
    parser.add_argument('--input', type=str, required=True, help='Input dataset directory')
    parser.add_argument('--output', type=str, help='Output directory for cleaned data')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, do not clean')
    parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224], 
                       help='Target image size (height width)')
    
    args = parser.parse_args()
    
    # Initialize cleaner
    cleaner = MedicalDataCleaner(args.input)
    
    # Analyze dataset
    print("🔍 Analyzing dataset quality...")
    quality_report = cleaner.analyze_dataset_quality()
    
    # Save quality report
    cleaner.create_data_summary("data_quality_issues.csv")
    
    if not args.analyze_only and args.output:
        # Clean dataset
        cleaned_path = cleaner.clean_dataset(
            args.output, 
            target_size=tuple(args.target_size)
        )
        print(f"\n✅ Cleaned dataset saved to: {cleaned_path}")
    
    return quality_report


if __name__ == "__main__":
    main()
