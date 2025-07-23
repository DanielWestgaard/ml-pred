import sys
import pandas as pd
import numpy as np

from model.model_factory.base_model import BaseModel
import utils.model.model_utils as model_utils
from utils.gpu_utils import get_gpu_config, log_gpu_status
from utils.logging_utils import get_logger

logger = get_logger()

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not installed. Install with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False


class TensorflowModel(BaseModel):
    """
    TensorFlow Neural Network Model with automatic GPU acceleration.
    
    Supports CUDA (NVIDIA) and Metal (Apple Silicon via tensorflow-metal).
    Automatically falls back to CPU if GPU is not available.
    """
    
    def __init__(self, dataset: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """Initialize TensorFlow model with GPU detection."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for this model. Install with: pip install tensorflow")
            
        logger.debug("----- Initializing TensorFlow Model -----")
        
        self.df = dataset
        self.X_train = X_train
        self.y_train = y_train
        
        # Get GPU configuration
        self.gpu_config = get_gpu_config('tensorflow')
        
        # Configure TensorFlow GPU settings
        self._configure_gpu()
        
        logger.info(f"TensorFlow device: {self.gpu_config['device']}")
        
    def _configure_gpu(self):
        """Configure TensorFlow GPU settings."""
        if 'memory_growth' in self.gpu_config and self.gpu_config['memory_growth']:
            # Enable memory growth to avoid allocating all GPU memory at once
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.debug("GPU memory growth enabled")
                except RuntimeError as e:
                    logger.warning(f"GPU memory growth configuration failed: {e}")
    
    def train(self):
        """Train TensorFlow neural network with GPU acceleration."""
        logger.info("---------- Starting to train TensorFlow Model! ----------")
        
        log_gpu_status()
        
        # Specify device context
        with tf.device(self.gpu_config['device']):
            # Prepare data
            X_train_scaled = self._scale_features(self.X_train)
            y_train_array = self.y_train.values.reshape(-1, 1)
            
            # Build neural network model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='linear')  # Regression output
            ])
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Training callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train_array,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1 if logger.level <= 20 else 0  # Verbose if DEBUG/INFO level
            )
            
            logger.info(f"Successfully trained TensorFlow Model using {self.gpu_config['device']}!")
            
            # Store training history
            model.training_history = history.history
            
            return model
    
    def _scale_features(self, X):
        """Scale features using StandardScaler-like normalization."""
        # Simple normalization - in production, use sklearn.preprocessing.StandardScaler
        X_scaled = (X - X.mean()) / X.std()
        return X_scaled.fillna(0)  # Handle any NaN values
    
    def save_model_and_info(self):
        """Save trained TensorFlow model."""
        pass