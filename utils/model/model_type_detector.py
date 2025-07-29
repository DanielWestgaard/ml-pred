# utils/model/model_type_detector.py
"""
Utilities for detecting and handling different model types across the trading system.
Supports both regression and classification models with automatic detection.
"""

from enum import Enum
from typing import Any, Dict, Optional, Union
import numpy as np
from pathlib import Path
import yaml

from utils.logging_utils import get_logger

logger = get_logger()

class ModelType(Enum):
    """Supported model types for the trading system"""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"

class ModelTypeDetector:
    """
    Detects model type from various sources: model object, metadata, or configuration.
    Provides utilities for handling different prediction formats.
    """
    
    @staticmethod
    def detect_from_model(model: Any) -> ModelType:
        """
        Auto-detect model type from model object
        
        Args:
            model: Trained model object
            
        Returns:
            Detected model type
        """
        model_class_name = model.__class__.__name__.lower()
        logger.debug(f"Detecting model type from class: {model.__class__.__name__}")
        
        # Classification model detection
        classification_keywords = ['classifier', 'classification']
        if any(keyword in model_class_name for keyword in classification_keywords):
            if hasattr(model, 'predict_proba'):
                # Check number of classes to distinguish binary vs multi-class
                if hasattr(model, 'classes_'):
                    n_classes = len(model.classes_)
                    if n_classes == 2:
                        logger.info(f"Detected binary classification model: {model.__class__.__name__} (2 classes)")
                        return ModelType.BINARY_CLASSIFICATION
                    elif n_classes > 2:
                        logger.info(f"Detected multi-class classification model: {model.__class__.__name__} ({n_classes} classes)")
                        return ModelType.MULTICLASS_CLASSIFICATION
                else:
                    logger.info(f"Detected binary classification model (assumed): {model.__class__.__name__}")
                    return ModelType.BINARY_CLASSIFICATION
            logger.info(f"Detected classification model (no probabilities): {model.__class__.__name__}")
            return ModelType.BINARY_CLASSIFICATION
        
        # Regression model detection
        regression_keywords = ['regressor', 'regression']
        if any(keyword in model_class_name for keyword in regression_keywords):
            logger.info(f"Detected regression model: {model.__class__.__name__}")
            return ModelType.REGRESSION
        
        # Fallback based on available methods
        if hasattr(model, 'predict_proba'):
            logger.info(f"Model has predict_proba, assuming binary classification: {model.__class__.__name__}")
            return ModelType.BINARY_CLASSIFICATION
        else:
            logger.info(f"Model lacks predict_proba, assuming regression: {model.__class__.__name__}")
            return ModelType.REGRESSION
    
    @staticmethod
    def detect_from_metadata(metadata_path: Union[str, Path]) -> Optional[ModelType]:
        """
        Detect model type from model metadata file
        
        Args:
            metadata_path: Path to model metadata YAML file
            
        Returns:
            Detected model type or None if not found
        """
        try:
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            
            # Check model_info section
            model_info = metadata.get('model_info', {})
            model_class = model_info.get('model_class', '').lower()
            
            if 'classifier' in model_class:
                logger.info(f"Detected classification model from metadata: {model_class}")
                return ModelType.BINARY_CLASSIFICATION
            elif 'regressor' in model_class:
                logger.info(f"Detected regression model from metadata: {model_class}")
                return ModelType.REGRESSION
            
            # Check dataset section for target information
            dataset_info = metadata.get('dataset', {})
            target_column = dataset_info.get('target_column', '').lower()
            
            if 'binary' in target_column or 'classification' in target_column:
                logger.info(f"Detected classification from target column: {target_column}")
                return ModelType.BINARY_CLASSIFICATION
            
            logger.warning(f"Could not determine model type from metadata at {metadata_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error reading metadata from {metadata_path}: {e}")
            return None
    
    @staticmethod
    def detect_from_config(config: Dict[str, Any]) -> Optional[ModelType]:
        """
        Detect model type from hypothesis configuration
        
        Args:
            config: Hypothesis configuration dictionary
            
        Returns:
            Detected model type or None if not found
        """
        # Check model training configuration
        model_config = config.get('model_training', {})
        model_type = model_config.get('model_type', '').lower()
        
        if 'classifier' in model_type or 'classification' in model_type:
            logger.info(f"Detected classification model from config: {model_type}")
            return ModelType.BINARY_CLASSIFICATION
        elif 'regressor' in model_type or 'regression' in model_type:
            logger.info(f"Detected regression model from config: {model_type}")
            return ModelType.REGRESSION
        
        # Check target configuration
        target_info = config.get('target', {})
        target_type = target_info.get('type', '').lower()
        
        if target_type in ['binary', 'classification']:
            logger.info(f"Detected classification from target type: {target_type}")
            return ModelType.BINARY_CLASSIFICATION
        elif target_type in ['regression', 'continuous']:
            logger.info(f"Detected regression from target type: {target_type}")
            return ModelType.REGRESSION
        
        logger.warning("Could not determine model type from configuration")
        return None

class PredictionProcessor:
    """
    Processes model predictions based on model type.
    Handles probability extraction and confidence calculation.
    """
    
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        logger.info(f"Prediction processor initialized for {model_type.value}")
    
    def process_predictions(self, model: Any, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process model predictions and extract relevant information
        
        Args:
            model: Trained model object
            features: Feature array for prediction
            
        Returns:
            Dictionary containing predictions, probabilities, and confidence scores
        """
        if self.model_type == ModelType.BINARY_CLASSIFICATION:
            return self._process_classification_predictions(model, features)
        elif self.model_type == ModelType.MULTICLASS_CLASSIFICATION:
            return self._process_multiclass_predictions(model, features)
        elif self.model_type == ModelType.REGRESSION:
            return self._process_regression_predictions(model, features)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _process_classification_predictions(self, model: Any, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Process binary classification predictions"""
        predictions = model.predict(features)
        
        # Get probabilities if available
        probabilities = None
        confidence_scores = None
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
            if probabilities.shape[1] == 2:  # Binary classification
                positive_probs = probabilities[:, 1]  # Probability of positive class
                confidence_scores = np.maximum(positive_probs, 1 - positive_probs)  # Distance from 0.5
            else:
                # Multi-class (future support)
                confidence_scores = np.max(probabilities, axis=1)
        else:
            # No probabilities available - use fixed confidence
            confidence_scores = np.where(predictions == 1, 0.8, 0.2)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'positive_probabilities': probabilities[:, 1] if probabilities is not None else None,
            'confidence_scores': confidence_scores
        }
    
    def _process_multiclass_predictions(self, model: Any, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Process multi-class classification predictions"""
        predictions = model.predict(features)
        
        # Get probabilities if available
        probabilities = None
        confidence_scores = None
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
            # For multi-class, confidence is the maximum probability across all classes
            confidence_scores = np.max(probabilities, axis=1)
        else:
            # No probabilities available - use fixed confidence based on predicted class
            # For three-class directional: higher confidence for non-hold classes
            confidence_scores = np.where(predictions == 0, 0.3, 0.7)  # HOLD=0.3, BUY/SELL=0.7
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'positive_probabilities': None,  # Not applicable for multi-class
            'confidence_scores': confidence_scores
        }
    
    def _process_regression_predictions(self, model: Any, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Process regression predictions"""
        predictions = model.predict(features)
        
        # For regression, confidence could be based on prediction magnitude or uncertainty
        # Simple approach: use distance from neutral (0.5) as confidence
        confidence_scores = np.abs(predictions - 0.5) * 2  # Normalize to 0-1
        
        return {
            'predictions': predictions,
            'probabilities': None,
            'positive_probabilities': None,
            'confidence_scores': confidence_scores
        }

def get_model_type(model: Any = None, metadata_path: Union[str, Path] = None, 
                  config: Dict[str, Any] = None) -> Optional[ModelType]:
    """
    Convenience function to detect model type from multiple sources
    
    Args:
        model: Model object (highest priority)
        metadata_path: Path to model metadata file
        config: Hypothesis configuration dictionary
        
    Returns:
        Detected model type or None if detection fails
    """
    detector = ModelTypeDetector()
    
    # Try model object first (most reliable)
    if model is not None:
        return detector.detect_from_model(model)
    
    # Try metadata file
    if metadata_path is not None:
        model_type = detector.detect_from_metadata(metadata_path)
        if model_type is not None:
            return model_type
    
    # Try configuration
    if config is not None:
        model_type = detector.detect_from_config(config)
        if model_type is not None:
            return model_type
    
    logger.error("Could not detect model type from any source")
    return None