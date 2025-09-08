"""
Training Module for Medical Computer Vision

This module provides training pipelines optimized for Dawit's system:
- CPU: Intel 10 cores (12 logical)
- RAM: 15.7 GB
- Optimal batch size: 8-16
- Training time: 4-12 hours per model
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
    TensorBoard, CSVLogger
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_models import create_model
from data.preprocessing import MedicalImagePreprocessor


class MedicalTrainer:
    """Training pipeline optimized for Dawit's laptop specs."""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.history = None
        
        # Optimize for Dawit's system
        self._optimize_for_system()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration from YAML file."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Override batch size for Dawit's system (15.7GB RAM, no GPU)
        if config['training']['batch_size'] > 16:
            print(f"🔧 Optimizing batch size for your system: {config['training']['batch_size']} → 16")
            config['training']['batch_size'] = 16
            
        return config
    
    def _optimize_for_system(self):
        """Optimize TensorFlow for Dawit's CPU-only system."""
        print("🔧 Optimizing TensorFlow for your system...")
        
        # Use all CPU cores efficiently
        tf.config.threading.set_intra_op_parallelism_threads(10)  # Physical cores
        tf.config.threading.set_inter_op_parallelism_threads(12)  # Logical cores
        
        # Optimize memory usage
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce verbose logging
        
        print(f"✓ Configured for {tf.config.threading.get_intra_op_parallelism_threads()} cores")
        print(f"✓ Batch size optimized: {self.config['training']['batch_size']}")
    
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare training data.
        
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        print("📁 Loading and preparing data...")
        
        data_path = Path(self.config['paths']['processed_data'])
        if not data_path.exists():
            print(f"❌ Processed data not found at {data_path}")
            print("Run preprocessing first: python src/data/preprocessing.py")
            sys.exit(1)
        
        # This is a placeholder - implement actual data loading
        # based on your dataset structure
        print("⚠️  Implement actual data loading in load_and_prepare_data()")
        
        # For now, create dummy data for testing
        img_size = self.config['preprocessing']['image_size']
        n_samples = 100  # Small for testing
        
        X = np.random.random((n_samples, img_size[0], img_size[1], 3)).astype(np.float32)
        y = np.random.randint(0, self.config['dataset']['num_classes'], n_samples)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config['training']['validation_split'],
            random_state=42,
            stratify=y
        )
        
        print(f"✓ Training samples: {len(X_train)}")
        print(f"✓ Validation samples: {len(X_val)}")
        print(f"✓ Classes: {self.config['dataset']['num_classes']}")
        
        return X_train, X_val, y_train, y_val
    
    def create_callbacks(self) -> list:
        """Create training callbacks optimized for Dawit's system."""
        callbacks = []
        
        # Create results directory
        results_dir = Path(self.config['paths']['logs'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping - save time on Dawit's CPU
        if self.config['training']['early_stopping']['enabled']:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.config['training']['early_stopping']['patience'],
                restore_best_weights=self.config['training']['early_stopping']['restore_best_weights'],
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Reduce learning rate
        if self.config['training']['reduce_lr']['enabled']:
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config['training']['reduce_lr']['factor'],
                patience=self.config['training']['reduce_lr']['patience'],
                min_lr=self.config['training']['reduce_lr']['min_lr'],
                verbose=1
            )
            callbacks.append(reduce_lr)
        
        # Model checkpoint
        checkpoint_path = Path(self.config['paths']['model_save'])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # CSV logger for tracking
        csv_logger = CSVLogger(
            str(results_dir / 'training_log.csv'),
            append=True
        )
        callbacks.append(csv_logger)
        
        print(f"✓ Callbacks configured for your system")
        return callbacks
    
    def train(self) -> keras.callbacks.History:
        """
        Train the model with system-specific optimizations.
        
        Returns:
            Training history
        """
        print("🚀 Starting training optimized for your laptop...")
        print(f"⏱️  Expected training time: 4-12 hours (CPU: 10 cores, RAM: 15.7GB)")
        
        # Load data
        X_train, X_val, y_train, y_val = self.load_and_prepare_data()
        
        # Create model
        print("🏗️  Creating model...")
        self.model = create_model(
            model_type=self.config['model']['type'],
            input_shape=tuple(self.config['model']['input_shape']),
            num_classes=self.config['dataset']['num_classes'],
            dropout_rate=self.config['model']['dropout_rate']
        )
        
        print(f"✓ Model created: {self.config['model']['type']}")
        print(f"✓ Parameters: {self.model.count_params():,}")
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        print(f"🎯 Training with batch size {self.config['training']['batch_size']} (optimized for your RAM)")
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['training']['batch_size'],
            epochs=self.config['training']['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate trained model.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        print("📊 Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test, batch_size=self.config['training']['batch_size'])
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        report = classification_report(
            y_test, y_pred_classes,
            target_names=self.config['dataset']['classes'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # Save results
        results_dir = Path(self.config['paths']['logs'])
        
        # Save classification report
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(results_dir / 'classification_report.csv')
        
        # Save confusion matrix
        cm_df = pd.DataFrame(cm, 
                           index=self.config['dataset']['classes'],
                           columns=self.config['dataset']['classes'])
        cm_df.to_csv(results_dir / 'confusion_matrix.csv')
        
        print("✅ Evaluation completed!")
        print(f"📄 Results saved to {results_dir}")
        
        return report


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train medical computer vision model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test with dummy data')
    
    args = parser.parse_args()
    
    print("🏥 Medical Computer Vision Training")
    print("=" * 50)
    print(f"👤 Training for: Dawit L. Gulta")
    print(f"💻 System: Intel 10-core CPU, 15.7GB RAM (CPU-only)")
    print(f"📁 Config: {args.config}")
    
    # Initialize trainer
    trainer = MedicalTrainer(args.config)
    
    if args.test:
        print("\n🧪 Running quick test...")
        # Override epochs for testing
        trainer.config['training']['epochs'] = 2
        print("✓ Set epochs to 2 for testing")
    
    # Train model
    history = trainer.train()
    
    print("\n📈 Training Summary:")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Best val_loss: {min(history.history['val_loss']):.4f}")
    print(f"Best val_accuracy: {max(history.history['val_accuracy']):.4f}")
    
    print(f"\n✅ Training completed successfully!")
    print(f"💾 Model saved to: {trainer.config['paths']['model_save']}")


if __name__ == "__main__":
    main()
