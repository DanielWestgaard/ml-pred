import logging
import os
import sys
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder
import yaml
from typing import Dict, Any, Optional, List
from datetime import datetime
import glob


def handle_categories(df:pd.DataFrame):
    """
    Simple method to handle columns of object type, and make them categorical type.
    The reason for this need, is that xgboost don't accept columns of object, so we need to
    change their type.
    
    Args:
        - df: the dataframe you intend to 
    """
    # list for keeping track of changed columns
    column_object_list = []
    
    # Loop through df to find columns of object type, and replace with categorical.
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].astype('category')
                column_object_list.append(col)
                logging.debug(f"Successfully changed {col} dtype from object to category!")
            except Exception as e:
                logging.error(f"Will drop {col}: Unable to change dtype of {col} from object to categorical: {e}")
                df = df.drop(col, axis=1)
    
    return df

def ensure_correct_xy_format(X:pd.DataFrame, y:pd.DataFrame):
    """
    Method used to check if the formats (dtype) is valid or not when using temporal_train_test_split()
    from sktime.split.
    
    Args: 
        - X: pd.DataFrame containing columns (Features).
        - y: pd.DataFrame containing the Target.
    """
    from sktime.datatypes import check_raise

    # Check current y format
    try:
        check_raise(y, mtype='pd.Series')
        logging.debug("✓ y is in correct format")
    except Exception as e:
        logging.ERROR(f"✗ Error with y: {e}")

    # Check current X format  
    try:
        check_raise(X, mtype='pd.DataFrame')
        logging.debug("✓ X is in correct format")
    except Exception as e:
        logging.error(f"✗ Error with X: {e}")

    # Let's also inspect your index
    logging.debug(f"Index type: {type(y.index)}")
    logging.debug(f"Index dtype: {y.index.dtype}")
    logging.debug(f"Index sample: {y.index[:5]}")
    logging.debug(f"Index frequency: {y.index.freq}")



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
        logging.error("Method not created yet! Exiting..")
        sys.exit()
    
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
    print("Hello!!")
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




# Deprecated?
def preprocess_features_for_xgboost(df, target_col=None, enable_categorical=False):
    """
    Preprocess features for XGBoost, handling all object and category types dynamically.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe with features
    target_col : str, optional
        Name of target column to exclude from processing
    enable_categorical : bool, default=False
        Whether to use XGBoost's native categorical support (requires XGBoost 1.5+)
        
    Returns:
    --------
    pandas DataFrame
        Processed dataframe ready for XGBoost
    """
    # First, check for and fix duplicate columns
    df_processed = df.copy()
    
    # 1. Handle fft_dominant_periods (list stored as string)
    if 'fft_dominant_periods' in df_processed.columns:
        try:
            # Convert string representation of list to actual values
            df_processed['fft_dominant_periods'] = df_processed['fft_dominant_periods'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            
            # Extract specific components (first three elements if available)
            max_components = 3
            for i in range(max_components):
                df_processed[f'fft_dom_period_{i+1}'] = df_processed['fft_dominant_periods'].apply(
                    lambda x: float(x[i]) if isinstance(x, (list, tuple)) and len(x) > i else np.nan
                )
            
            # Calculate mean and max of components as additional features
            df_processed['fft_dom_period_mean'] = df_processed['fft_dominant_periods'].apply(
                lambda x: np.mean(x) if isinstance(x, (list, tuple)) and len(x) > 0 else np.nan
            )
            df_processed['fft_dom_period_max'] = df_processed['fft_dominant_periods'].apply(
                lambda x: np.max(x) if isinstance(x, (list, tuple)) and len(x) > 0 else np.nan
            )
            
            # Drop the original column
            df_processed = df_processed.drop('fft_dominant_periods', axis=1)
        
        except Exception as e:
            print(f"Error processing fft_dominant_periods: {e}")
            # Fallback: Label encode as a last resort
            le = LabelEncoder()
            df_processed['fft_dom_encoded'] = le.fit_transform(df_processed['fft_dominant_periods'])
            df_processed = df_processed.drop('fft_dominant_periods', axis=1)
    
    # 2. Process other categorical features
    cat_columns = []
    for col in df_processed.columns:
        if col == target_col:
            continue
        # Check if it's a single column (not duplicated)
        if isinstance(df_processed[col], pd.Series):
            if df_processed[col].dtype == 'object' or df_processed[col].dtype.name == 'category':
                cat_columns.append(col)
    
    if enable_categorical:
        # Option 1: Use XGBoost's native categorical support
        for col in cat_columns:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].astype('category')
    else:
        # Option 2: Encode categorical features
        for col in cat_columns:
            n_unique = df_processed[col].nunique()
            
            if n_unique <= 10:  # Low cardinality -> one-hot encoding
                # Create dummies with better column names for clarity
                dummies = pd.get_dummies(
                    df_processed[col], 
                    prefix=col, 
                    prefix_sep='_',
                    drop_first=True  # Remove one category to avoid multicollinearity
                )
                df_processed = pd.concat([df_processed, dummies], axis=1)
            else:  # High cardinality -> label encoding
                le = LabelEncoder()
                df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            
            # Drop original categorical column
            df_processed = df_processed.drop(col, axis=1)
    
    # 3. Final check for any remaining non-numeric columns
    non_numeric_cols = []
    for col in df_processed.columns:
        if col == target_col:
            continue
        # Check if it's a single column (not duplicated)
        try:
            is_numeric = pd.api.types.is_numeric_dtype(df_processed[col])
            if not is_numeric:
                non_numeric_cols.append(col)
        except Exception as e:
            print(f"Error checking column {col}: {e}")
            non_numeric_cols.append(col)
    
    # Drop non-numeric columns
    if non_numeric_cols:
        print(f"Warning: Dropping non-numeric columns: {non_numeric_cols}")
        df_processed = df_processed.drop(non_numeric_cols, axis=1)
    
    return df_processed