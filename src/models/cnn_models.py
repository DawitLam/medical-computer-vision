"""
CNN Models for Medical Image Classification

This module contains custom CNN architectures and transfer learning models
specifically designed for medical image analysis.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    ResNet50, VGG16, DenseNet121, EfficientNetB0, 
    InceptionV3, Xception
)
import numpy as np
from typing import Tuple, Optional, List


class MedicalCNN:
    """Custom CNN architectures for medical image classification."""
    
    @staticmethod
    def build_basic_cnn(input_shape: Tuple[int, int, int], 
                       num_classes: int, 
                       dropout_rate: float = 0.5) -> Model:
        """
        Build a basic CNN architecture.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=input_shape)
        
        # First conv block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate * 0.25)(x)
        
        # Second conv block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate * 0.25)(x)
        
        # Third conv block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        
        # Fourth conv block
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        
        # Global average pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='basic_medical_cnn')
        return model
    
    @staticmethod
    def build_residual_cnn(input_shape: Tuple[int, int, int], 
                          num_classes: int,
                          dropout_rate: float = 0.5) -> Model:
        """
        Build a CNN with residual connections.
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=input_shape)
        
        # Initial conv layer
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        # Residual blocks
        for filters in [64, 128, 256, 512]:
            # First block (with downsampling if not first)
            shortcut = x
            if filters != 64:
                x = layers.Conv2D(filters, (1, 1), strides=2, padding='same')(x)
                shortcut = layers.Conv2D(filters, (1, 1), strides=2, padding='same')(shortcut)
            else:
                x = layers.Conv2D(filters, (1, 1), padding='same')(x)
                shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
            
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout_rate * 0.25)(x)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='residual_medical_cnn')
        return model
    
    @staticmethod
    def build_attention_cnn(input_shape: Tuple[int, int, int], 
                           num_classes: int,
                           dropout_rate: float = 0.5) -> Model:
        """
        Build a CNN with attention mechanisms.
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=input_shape)
        
        # Feature extraction
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        features = layers.MaxPooling2D((2, 2))(x)
        
        # Attention mechanism
        attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(features)
        attended_features = layers.Multiply()([features, attention])
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling2D()(attended_features)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='attention_medical_cnn')
        return model


class MedicalTransferLearning:
    """Transfer learning models for medical image classification."""
    
    @staticmethod
    def build_transfer_model(base_model_name: str,
                           input_shape: Tuple[int, int, int],
                           num_classes: int,
                           freeze_base: bool = True,
                           dropout_rate: float = 0.5) -> Model:
        """
        Build a transfer learning model.
        
        Args:
            base_model_name: Name of the base model
            input_shape: Input image shape
            num_classes: Number of output classes
            freeze_base: Whether to freeze base model weights
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        # Get base model
        base_models = {
            'resnet50': ResNet50,
            'vgg16': VGG16,
            'densenet121': DenseNet121,
            'efficientnetb0': EfficientNetB0,
            'inceptionv3': InceptionV3,
            'xception': Xception
        }
        
        if base_model_name.lower() not in base_models:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        base_model_class = base_models[base_model_name.lower()]
        
        # Handle input shape for models that require specific sizes
        if base_model_name.lower() in ['inceptionv3', 'xception']:
            if input_shape[0] < 75:
                input_shape = (150, 150, input_shape[2])
        
        # Create base model
        base_model = base_model_class(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze base model if specified
        if freeze_base:
            base_model.trainable = False
        
        # Add custom classification head
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name=f'{base_model_name}_medical')
        return model
    
    @staticmethod
    def unfreeze_top_layers(model: Model, num_layers: int = 10) -> Model:
        """
        Unfreeze top layers of a transfer learning model for fine-tuning.
        
        Args:
            model: Transfer learning model
            num_layers: Number of top layers to unfreeze
            
        Returns:
            Model with unfrozen layers
        """
        # Find the base model (usually the first layer)
        base_model = None
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                base_model = layer
                break
        
        if base_model is None:
            print("Could not find base model for unfreezing")
            return model
        
        # Unfreeze top layers
        for layer in base_model.layers[-num_layers:]:
            layer.trainable = True
        
        print(f"Unfroze top {num_layers} layers")
        return model


class Model3D:
    """3D CNN models for volumetric medical data."""
    
    @staticmethod
    def build_3d_cnn(input_shape: Tuple[int, int, int, int],
                     num_classes: int,
                     dropout_rate: float = 0.5) -> Model:
        """
        Build a 3D CNN for volumetric data.
        
        Args:
            input_shape: Input volume shape (depth, height, width, channels)
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=input_shape)
        
        # First 3D conv block
        x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        x = layers.Dropout(dropout_rate * 0.25)(x)
        
        # Second 3D conv block
        x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        x = layers.Dropout(dropout_rate * 0.25)(x)
        
        # Third 3D conv block
        x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='3d_medical_cnn')
        return model


def compile_model(model: Model, 
                 num_classes: int,
                 learning_rate: float = 0.001,
                 metrics: Optional[List[str]] = None) -> Model:
    """
    Compile a model with appropriate loss function and metrics.
    
    Args:
        model: Keras model to compile
        num_classes: Number of classes
        learning_rate: Learning rate for optimizer
        metrics: List of metrics to track
        
    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = ['accuracy']
    
    # Choose appropriate loss function
    if num_classes == 2:
        loss = 'binary_crossentropy'
        if 'accuracy' not in metrics:
            metrics.append('accuracy')
    else:
        loss = 'sparse_categorical_crossentropy'
        if 'sparse_categorical_accuracy' not in metrics:
            metrics.append('sparse_categorical_accuracy')
    
    # Add medical-specific metrics
    medical_metrics = ['precision', 'recall']
    for metric in medical_metrics:
        if metric not in metrics:
            metrics.append(metric)
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


# Model factory function
def create_model(model_type: str,
                input_shape: Tuple[int, int, int],
                num_classes: int,
                **kwargs) -> Model:
    """
    Factory function to create different model types.
    
    Args:
        model_type: Type of model to create
        input_shape: Input image shape
        num_classes: Number of classes
        **kwargs: Additional arguments
        
    Returns:
        Compiled Keras model
    """
    if model_type == 'basic_cnn':
        model = MedicalCNN.build_basic_cnn(input_shape, num_classes, **kwargs)
    elif model_type == 'residual_cnn':
        model = MedicalCNN.build_residual_cnn(input_shape, num_classes, **kwargs)
    elif model_type == 'attention_cnn':
        model = MedicalCNN.build_attention_cnn(input_shape, num_classes, **kwargs)
    elif model_type.startswith('transfer_'):
        base_model_name = model_type.replace('transfer_', '')
        model = MedicalTransferLearning.build_transfer_model(
            base_model_name, input_shape, num_classes, **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return compile_model(model, num_classes)
