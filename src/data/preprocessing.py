"""
Image Preprocessing Module for Medical CT Images

This module provides comprehensive preprocessing utilities for medical images
including resizing, normalization, windowing, and DICOM handling.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Union
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import yaml


class MedicalImagePreprocessor:
    """Handles preprocessing of medical images for ML pipelines."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path) if config_path else self._default_config()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _default_config(self) -> dict:
        """Default preprocessing configuration."""
        return {
            'image_size': (224, 224),
            'normalize': True,
            'hu_window': {'level': 40, 'width': 400},  # Lung window
            'pixel_spacing': None,
            'output_format': 'png'
        }
    
    def load_dicom(self, dicom_path: str) -> Tuple[np.ndarray, dict]:
        """
        Load DICOM file and extract metadata.
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            Tuple of (pixel array, metadata dictionary)
        """
        ds = pydicom.dcmread(dicom_path)
        
        # Extract pixel array
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Apply rescale slope and intercept
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        # Extract metadata
        metadata = {
            'patient_id': getattr(ds, 'PatientID', 'Unknown'),
            'study_date': getattr(ds, 'StudyDate', 'Unknown'),
            'modality': getattr(ds, 'Modality', 'Unknown'),
            'slice_thickness': getattr(ds, 'SliceThickness', None),
            'pixel_spacing': getattr(ds, 'PixelSpacing', None),
            'window_center': getattr(ds, 'WindowCenter', None),
            'window_width': getattr(ds, 'WindowWidth', None)
        }
        
        return pixel_array, metadata
    
    def apply_hu_windowing(self, image: np.ndarray, level: int, width: int) -> np.ndarray:
        """
        Apply Hounsfield Unit windowing to CT images.
        
        Args:
            image: Input image array
            level: Window level (center)
            width: Window width
            
        Returns:
            Windowed image array
        """
        min_hu = level - width // 2
        max_hu = level + width // 2
        
        # Clip values to window range
        windowed = np.clip(image, min_hu, max_hu)
        
        # Normalize to 0-255 range
        windowed = ((windowed - min_hu) / (max_hu - min_hu) * 255).astype(np.uint8)
        
        return windowed
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image array
            target_size: Target (height, width)
            
        Returns:
            Resized image array
        """
        if len(image.shape) == 2:
            return cv2.resize(image, target_size[::-1])  # cv2 uses (width, height)
        else:
            return cv2.resize(image, target_size[::-1])
    
    def normalize_image(self, image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize image values.
        
        Args:
            image: Input image array
            method: Normalization method ('minmax', 'zscore', 'unit')
            
        Returns:
            Normalized image array
        """
        if method == 'minmax':
            return (image - image.min()) / (image.max() - image.min())
        elif method == 'zscore':
            return (image - image.mean()) / image.std()
        elif method == 'unit':
            return image / 255.0
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Enhance image contrast.
        
        Args:
            image: Input image array
            method: Enhancement method ('clahe', 'histogram_eq')
            
        Returns:
            Enhanced image array
        """
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image.astype(np.uint8))
        elif method == 'histogram_eq':
            return cv2.equalizeHist(image.astype(np.uint8))
        else:
            raise ValueError(f"Unknown enhancement method: {method}")
    
    def remove_noise(self, image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        """
        Remove noise from image.
        
        Args:
            image: Input image array
            method: Denoising method ('gaussian', 'median', 'bilateral')
            
        Returns:
            Denoised image array
        """
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
    
    def process_single_image(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save processed image
            
        Returns:
            Processed image array
        """
        # Load image
        if image_path.lower().endswith('.dcm'):
            image, metadata = self.load_dicom(image_path)
            
            # Apply HU windowing if CT scan
            if metadata.get('modality') == 'CT':
                image = self.apply_hu_windowing(
                    image, 
                    self.config['hu_window']['level'],
                    self.config['hu_window']['width']
                )
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        
        # Resize
        image = self.resize_image(image, self.config['image_size'])
        
        # Enhance contrast
        image = self.enhance_contrast(image)
        
        # Remove noise
        image = self.remove_noise(image)
        
        # Normalize
        if self.config['normalize']:
            image = self.normalize_image(image, 'unit')
        
        # Save if output path provided
        if output_path:
            self.save_processed_image(image, output_path)
        
        return image
    
    def process_dataset(self, input_dir: str, output_dir: str, 
                       file_extension: str = '.png') -> pd.DataFrame:
        """
        Process an entire dataset.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for processed images
            file_extension: File extension for output images
            
        Returns:
            DataFrame with processing metadata
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.dcm', '.dicom']:
            image_files.extend(input_path.glob(f'**/*{ext}'))
        
        processed_files = []
        
        for img_file in image_files:
            try:
                # Create output path maintaining directory structure
                rel_path = img_file.relative_to(input_path)
                output_file = output_path / rel_path.with_suffix(file_extension)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Process image
                processed_img = self.process_single_image(str(img_file))
                
                # Save processed image
                self.save_processed_image(processed_img, str(output_file))
                
                processed_files.append({
                    'original_path': str(img_file),
                    'processed_path': str(output_file),
                    'status': 'success'
                })
                
                print(f"Processed: {img_file.name}")
                
            except Exception as e:
                processed_files.append({
                    'original_path': str(img_file),
                    'processed_path': None,
                    'status': f'error: {str(e)}'
                })
                print(f"Error processing {img_file.name}: {e}")
        
        return pd.DataFrame(processed_files)
    
    def save_processed_image(self, image: np.ndarray, output_path: str) -> None:
        """
        Save processed image to file.
        
        Args:
            image: Processed image array
            output_path: Output file path
        """
        # Convert to 8-bit if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        # Save image
        cv2.imwrite(output_path, image)
    
    def visualize_preprocessing_steps(self, image_path: str) -> None:
        """
        Visualize preprocessing steps for a single image.
        
        Args:
            image_path: Path to input image
        """
        # Load original image
        if image_path.lower().endswith('.dcm'):
            original, _ = self.load_dicom(image_path)
        else:
            original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply steps gradually
        resized = self.resize_image(original, self.config['image_size'])
        enhanced = self.enhance_contrast(resized)
        denoised = self.remove_noise(enhanced)
        normalized = self.normalize_image(denoised, 'unit')
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        images = [original, resized, enhanced, denoised, normalized]
        titles = ['Original', 'Resized', 'Enhanced', 'Denoised', 'Normalized']
        
        for i, (img, title) in enumerate(zip(images, titles)):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')
        
        # Hide the last subplot
        axes[5].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess medical images')
    parser.add_argument('--input', type=str, required=True, help='Input directory or file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--visualize', type=str, help='Visualize preprocessing for a single image')
    
    args = parser.parse_args()
    
    preprocessor = MedicalImagePreprocessor(args.config)
    
    if args.visualize:
        preprocessor.visualize_preprocessing_steps(args.visualize)
    elif os.path.isfile(args.input):
        # Process single file
        preprocessor.process_single_image(args.input, args.output)
    else:
        # Process directory
        df = preprocessor.process_dataset(args.input, args.output)
        print(f"Processed {len(df)} files")
        print(f"Success: {len(df[df['status'] == 'success'])}")
        print(f"Errors: {len(df[df['status'] != 'success'])}")


if __name__ == "__main__":
    main()
