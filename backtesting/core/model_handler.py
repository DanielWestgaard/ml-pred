import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from utils.logging_utils import get_logger
from utils.model.model_metadata_loader import load_model_metadata
from utils.model.model_type_detector import ModelTypeDetector, PredictionProcessor, get_model_type, ModelType

logger = get_logger()

class BacktestModelHandler:
    """
    Handles model loading, prediction processing, and model-related operations
    for backtesting. Separates model concerns from the main orchestration logic.
    """
    
    def __init__(self, config: Dict[str, Any], model_folder: Optional[str] = None):
        """
        Initialize model handler
        
        Args:
            config: Hypothesis configuration
            model_folder: Optional model folder path override
        """
        self.config = config
        self.model_folder = model_folder
        
        # Model components
        self.model = None
        self.model_metadata = None
        self.model_type = None
        self.prediction_processor = None
        
        # Load model components
        self._determine_model_path()
        self._load_model_and_metadata()
        self._detect_and_setup_model_type()
        
        logger.info("BacktestModelHandler initialized successfully")
    
    def _determine_model_path(self):
        """Determine the path to the model to load"""
        # Priority order: 1) explicit parameter, 2) config sections, 3) auto-discovery
        if self.model_folder:
            self.model_path = self.model_folder
            return
        
        # Check different config sections
        backtest_config = self.config.get('backtesting', {})
        model_training_config = self.config.get('model_training', {})
        strategy_config = self.config.get('strategy', {})
        
        self.model_path = (
            backtest_config.get('model_folder') or
            model_training_config.get('model_folder') or
            strategy_config.get('model_folder')
        )
        
        if not self.model_path:
            # Auto-discovery as last resort
            self.model_path = self._find_latest_model()
        
        if not self.model_path:
            raise ValueError("No model path specified and no trained models found")
        
        logger.info(f"Using model path: {self.model_path}")
    
    def _find_latest_model(self) -> Optional[str]:
        """Find the latest model for the hypothesis (fallback method)"""
        try:
            from utils.model.model_metadata_loader import get_latest_model
            hypothesis_name = self.config.get('hypothesis_name', 'spread_compression_patterns')
            return get_latest_model("storage/models", model_type=None)
        except ImportError:
            return self._legacy_find_latest_model()
    
    def _legacy_find_latest_model(self) -> Optional[str]:
        """Legacy method for finding latest model"""
        models_dir = Path("storage/models")
        if not models_dir.exists():
            return None
        
        hypothesis_name = self.config.get('hypothesis_name', '')
        matching_models = []
        
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "model_metadata.yml"
                if metadata_path.exists():
                    try:
                        import yaml
                        with open(metadata_path, 'r') as f:
                            metadata = yaml.safe_load(f)
                        if metadata.get('hypothesis', {}).get('name') == hypothesis_name:
                            matching_models.append(model_dir)
                    except Exception:
                        continue
        
        if matching_models:
            latest_model = sorted(matching_models, key=lambda x: x.name)[-1]
            return str(latest_model)
        
        return None
    
    def _load_model_and_metadata(self):
        """Load the model and its metadata"""
        try:
            # Load metadata using existing utility
            loader = load_model_metadata(self.model_path)
            self.model_metadata = loader.get_full_metadata()
            
            # Load the actual model
            model_type = loader.get_model_type()
            model_file = Path(self.model_path) / f"{model_type}.pkl"
            
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f"Loaded model: {model_type} from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _detect_and_setup_model_type(self):
        """Detect model type and set up prediction processor"""
        try:
            # Try to detect model type using metadata and model object
            metadata_path = Path(self.model_path) / 'model_metadata.yml'
            
            self.model_type = get_model_type(
                model=self.model,
                metadata_path=metadata_path,
                config=self.config
            )
            
            if self.model_type is None:
                # Fallback to direct detection
                detector = ModelTypeDetector()
                self.model_type = detector.detect_from_model(self.model)
            
            # Initialize prediction processor
            self.prediction_processor = PredictionProcessor(self.model_type)
            
            logger.info(f"Model type detected: {self.model_type.value}")
            
        except Exception as e:
            logger.error(f"Error detecting model type: {e}")
            # Fallback to regression
            self.model_type = ModelType.REGRESSION
            self.prediction_processor = PredictionProcessor(self.model_type)
            logger.warning("Defaulting to regression model type")
    
    def get_required_features(self) -> List[str]:
        """
        Get the features required by the model for predictions
        
        Returns:
            List of feature names required by the model
        """
        # Try to get from model object first
        if hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        
        # Fall back to metadata
        model_features = self.model_metadata.get('dataset', {}).get('features', [])
        target_column = self.model_metadata.get('dataset', {}).get('target_column', '')
        
        # Filter out target column from features
        required_features = [f for f in model_features if f != target_column]
        
        logger.debug(f"Required features: {len(required_features)} features")
        return required_features
    
    def make_predictions(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions using the loaded model
        
        Args:
            features: Feature array for prediction
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        try:
            # Use the prediction processor to handle different model types
            prediction_results = self.prediction_processor.process_predictions(
                self.model, features
            )
            
            predictions = prediction_results['predictions']
            
            # Get appropriate probabilities based on model type
            if self.model_type.value == 'multiclass_classification':
                probabilities = prediction_results.get('probabilities')
            else:
                probabilities = prediction_results.get('positive_probabilities')
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def validate_features(self, feature_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that all required features are present in the data
        
        Args:
            feature_data: Dictionary of available features
            
        Returns:
            Tuple of (is_valid, missing_features)
        """
        required_features = self.get_required_features()
        available_features = list(feature_data.keys())
        
        missing_features = [f for f in required_features if f not in available_features]
        is_valid = len(missing_features) == 0
        
        if not is_valid:
            logger.debug(f"Missing features: {missing_features[:5]}...")  # Show first 5
        
        return is_valid, missing_features
    
    def prepare_features_for_prediction(self, data_row) -> Optional[np.ndarray]:
        """
        Prepare features from a data row for prediction
        
        Args:
            data_row: pandas Series or dict with feature data
            
        Returns:
            Feature array ready for prediction or None if invalid
        """
        try:
            required_features = self.get_required_features()
            
            # Check if all required features are available
            if hasattr(data_row, 'index'):
                # pandas Series
                available_features = data_row.index.tolist()
                missing_features = [f for f in required_features if f not in available_features]
            else:
                # dict-like
                available_features = list(data_row.keys())
                missing_features = [f for f in required_features if f not in available_features]
            
            if missing_features:
                logger.debug(f"Cannot make prediction - missing features: {missing_features[:3]}...")
                return None
            
            # Extract features in the correct order
            if hasattr(data_row, 'loc'):
                # pandas Series with .loc
                feature_values = data_row[required_features].values
            else:
                # dict-like access
                feature_values = [data_row[f] for f in required_features]
            
            # Reshape for single prediction
            features = np.array(feature_values).reshape(1, -1)
            
            return features
            
        except Exception as e:
            logger.debug(f"Error preparing features: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': self.model_path,
            'model_class': self.model.__class__.__name__,
            'model_type': self.model_type.value if self.model_type else 'unknown',
            'hypothesis_name': self.model_metadata.get('hypothesis', {}).get('name', 'unknown'),
            'training_date': self.model_metadata.get('model_info', {}).get('training_date'),
            'feature_count': len(self.get_required_features()),
            'has_probabilities': hasattr(self.model, 'predict_proba'),
            'model_metadata': self.model_metadata
        }
    
    def get_data_source_info(self) -> Dict[str, Any]:
        """
        Get information about the expected data source from model metadata
        
        Returns:
            Dictionary with data source information
        """
        dataset_info = self.model_metadata.get('dataset', {})
        
        return {
            'dataset_path': dataset_info.get('dataset_path'),
            'features': dataset_info.get('features', []),
            'target_column': dataset_info.get('target_column'),
            'feature_count': len(dataset_info.get('features', [])),
            'training_period': dataset_info.get('training_period')
        }