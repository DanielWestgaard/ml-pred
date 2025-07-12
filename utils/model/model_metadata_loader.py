import logging
import os
import yaml
from typing import Dict, Any, Optional, List
from datetime import datetime


########## Code for loading contents of model_metadata.yml - metadata related to trained model ##########

class ModelMetadataLoader:
    """Utility class for loading and accessing model metadata from YAML files."""
    
    def __init__(self, metadata_path: str):
        """
        Initialize with path to metadata file or model directory.
        
        Args:
            metadata_path: Path to model_metadata.yml file or directory containing it
        """
        if os.path.isdir(metadata_path):
            # If directory provided, look for metadata file
            metadata_file = os.path.join(metadata_path, 'model_metadata.yml')
        else:
            metadata_file = metadata_path
            
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
        self.metadata_path = metadata_file
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from YAML file."""
        try:
            with open(self.metadata_path, 'r') as file:
                metadata = yaml.safe_load(file)
            logging.debug(f"Successfully loaded metadata from {self.metadata_path}")
            return metadata
        except Exception as e:
            logging.error(f"Error loading metadata: {e}")
            raise
    
    # General access methods
    def get_full_metadata(self) -> Dict[str, Any]:
        """Return complete metadata dictionary."""
        return self.metadata
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """Get specific section from metadata."""
        return self.metadata.get(section_name, {})
    
    def get_value(self, section: str, key: str, default=None):
        """Get specific value from metadata."""
        return self.metadata.get(section, {}).get(key, default)
    
    # Model info methods
    def get_model_type(self) -> str:
        """Get model type."""
        return self.get_value('model_info', 'model_type', 'unknown')
    
    def get_model_class(self) -> str:
        """Get model class name."""
        return self.get_value('model_info', 'model_class', 'unknown')
    
    def get_model_path(self) -> str:
        """Get storage path of the model."""
        return self.get_value('model_info', 'storage_path', '')
    
    def get_creation_date(self) -> Optional[datetime]:
        """Get model creation date as datetime object."""
        created_at = self.get_value('model_info', 'created_at')
        if created_at:
            try:
                return datetime.fromisoformat(created_at)
            except ValueError:
                logging.warning(f"Could not parse creation date: {created_at}")
        return None
    
    # Training info methods
    def get_training_duration(self) -> float:
        """Get training duration in seconds."""
        return self.get_value('training', 'duration_seconds', 0.0)
    
    def get_training_samples(self) -> tuple:
        """Get (train_samples, test_samples)."""
        train = self.get_value('training', 'train_samples', 0)
        test = self.get_value('training', 'test_samples', 0)
        return train, test
    
    # Dataset info methods
    def get_dataset_path(self) -> str:
        """Get dataset path."""
        return self.get_value('dataset', 'dataset_path', 'unknown')
    
    def get_target_column(self) -> str:
        """Get target column name."""
        return self.get_value('dataset', 'target_column', 'unknown')
    
    def get_features(self) -> List[str]:
        """Get list of feature columns."""
        return self.get_value('dataset', 'features', [])
    
    def get_feature_count(self) -> int:
        """Get number of features."""
        return self.get_value('dataset', 'feature_count', 0)
    
    def get_date_range(self) -> tuple:
        """Get (start_date, end_date) of training data."""
        date_range = self.get_value('dataset', 'date_range', {})
        return date_range.get('start'), date_range.get('end')
    
    # Model parameters methods
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return self.get_section('model_parameters')
    
    def get_parameter(self, param_name: str, default=None):
        """Get specific model parameter."""
        return self.get_value('model_parameters', param_name, default)
    
    # Hypothesis methods
    def get_hypothesis_type(self):
        return self.get_value('hypothesis', 'name')
    
    # System info methods
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return self.get_section('system_info')
    
    def get_hardware_specs(self) -> str:
        """Get formatted hardware specifications."""
        sys_info = self.get_system_info()
        cpu = sys_info.get('cpu', 'Unknown')
        memory = sys_info.get('memory_gb', 'Unknown')
        return f"CPU: {cpu}, Memory: {memory}GB"
    
    # Utility methods
    def print_summary(self):
        """Print a formatted summary of the model."""
        print("=" * 50)
        print(f"Model: {self.get_model_type()}")
        print(f"Created: {self.get_creation_date()}")
        print(f"Target: {self.get_target_column()}")
        print(f"Features: {self.get_feature_count()}")
        train, test = self.get_training_samples()
        print(f"Training samples: {train}, Test samples: {test}")
        print(f"Training duration: {self.get_training_duration():.2f}s")
        print(f"Hardware: {self.get_hardware_specs()}")
        print("=" * 50)
    
    def compare_with(self, other_metadata_path: str):
        """Compare this model with another model's metadata."""
        other = ModelMetadataLoader(other_metadata_path)
        
        print("Model Comparison")
        print("=" * 60)
        print(f"{'Metric':<20} {'Model 1':<15} {'Model 2':<15}")
        print("-" * 60)
        print(f"{'Model Type':<20} {self.get_model_type():<15} {other.get_model_type():<15}")
        print(f"{'Features':<20} {self.get_feature_count():<15} {other.get_feature_count():<15}")
        
        train1, test1 = self.get_training_samples()
        train2, test2 = other.get_training_samples()
        print(f"{'Train Samples':<20} {train1:<15} {train2:<15}")
        print(f"{'Test Samples':<20} {test1:<15} {test2:<15}")
        print(f"{'Duration (s)':<20} {self.get_training_duration():<15.2f} {other.get_training_duration():<15.2f}")


# Standalone utility functions
def load_model_metadata(metadata_path: str) -> ModelMetadataLoader:
    """
    Convenience function to load model metadata.
    
    Args:
        metadata_path: Path to metadata file or model directory
        
    Returns:
        ModelMetadataLoader instance
    """
    return ModelMetadataLoader(metadata_path)


def find_models_by_type(models_directory: str, model_type: str) -> List[str]:
    """
    Find all models of specific type in models directory.
    
    Args:
        models_directory: Base directory containing model folders
        model_type: Type of model to search for
        
    Returns:
        List of paths to model directories
    """
    model_dirs = []
    
    # Search through all subdirectories
    for root, dirs, files in os.walk(models_directory):
        if 'model_metadata.yml' in files:
            try:
                loader = ModelMetadataLoader(root)
                if loader.get_model_type() == model_type:
                    model_dirs.append(root)
            except Exception as e:
                logging.warning(f"Could not load metadata from {root}: {e}")
    
    return model_dirs

def get_latest_model(models_directory: str, model_type: str = None) -> Optional[str]:
    """
    Get the most recently created model.
    
    Args:
        models_directory: Base directory containing model folders
        model_type: Optional filter by model type
        
    Returns:
        Path to latest model directory, or None if no models found
    """
    
    latest_model = None
    latest_date = None
    
    for root, dirs, files in os.walk(models_directory):
        if 'model_metadata.yml' in files:
            try:
                loader = ModelMetadataLoader(root)
                
                # Filter by model type if specified
                if model_type and loader.get_model_type() != model_type:
                    continue
                
                creation_date = loader.get_creation_date()
                if creation_date and (latest_date is None or creation_date > latest_date):
                    latest_date = creation_date
                    latest_model = root
                    
            except Exception as e:
                logging.warning(f"Could not load metadata from {root}: {e}")
    
    logging.debug(f"Utility function, get_latest_model(), found the latest trained model: {latest_model}")
    return latest_model