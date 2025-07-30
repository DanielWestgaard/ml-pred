import datetime
import importlib
import os
import sys
import numpy as np
import pandas as pd
import pickle
import psutil
import yaml  # stroring
import time
import platform

# Train-test
from sktime.split import temporal_train_test_split  # https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.split.temporal_train_test_split.html

from utils import data_utils, shared_utils
import config.config as config
from utils.model import model_utils
import utils.hypothesis.config_loader as config_loader
from utils.logging_utils import get_logger

logger = get_logger()


class ModelTraining():
    """Train a specific ML Model."""
    def __init__(self, dataset:str = None, model_type:str = None, target_col:str = 'close', hypothesis_source:str = None, use_walk_forward:bool = None):
        """
        Initialize the class by providing specifics.
        
        Args:
            - dataset: Path to the Engineered (ohlcv) dataset to train the model on. Is 'optional' as it might be provided in hypothesis_source
            - model_type: Type of model you want to train. Eg. 'xgboost', 'decision_tree', 'random_forest', etc. Is 'optional' as it might be provided in hypothesis_source
            - TODO: hypothesis: Provide path to hypothesis' config.yml to target its column. This currently only works close target.
        """
        try:
            logger.info(f"---------- Initiating Model Training ----------")
            
            self.hypothesis_source = hypothesis_source
            # Hypothesis
            if self.hypothesis_source is not None:
                logger.info(f"Model Training was initialized with Hypothesis configuration!")
                
                # Loading in all configurations in the provided config.yml
                configurations = config_loader.load_hypothesis_config(hypothesis_source).handle_hypothesis_source()
                
                self.model_type = configurations.get("model_training_type") or configurations.get("model_type")
                
                # Extract target column configuration (proper fix)
                self.configured_target = configurations.get("model_training_target_column")
                # exclude basic OHLC features while keeping enhanced features
                self.exclude_basic_ohlc = configurations.get("model_training_exclude_basic_ohlc")
                
                # Extract walk-forward validation configuration
                self.walk_forward_config = configurations.get("walk_forward", {})
                if use_walk_forward is None:
                    # Auto-detect: use walk-forward if config exists or for time series data
                    self.use_walk_forward = bool(self.walk_forward_config) or self._is_time_series_data()
                else:
                    self.use_walk_forward = use_walk_forward
                
                try:
                    dataset = configurations.get("model_training_processed_data")
                    
                    if dataset is None:
                        raw_dataset = configurations.get("data_pipeline_raw_source")                        
                        
                        if raw_dataset is None:
                            raw_dataset = configurations.get("raw_data_source")

                        if raw_dataset is not None:
                            dataset = self._get_processed_path(raw_dataset)
                        else:
                            raise ValueError("No raw data source found")
                except:  # Failed means user either forgot to provide datapath, og have ran data pipeline first, meaning we must find that one
                    try:
                        raw_dataset = configurations.get("data_pipeline_raw_source")
                        
                        if raw_dataset is None:
                            raw_dataset = configurations.get("raw_data_source")
                            
                        if raw_dataset is not None:
                            dataset = self._get_processed_path(raw_dataset)
                        else:
                            raise ValueError("No raw data source found")
                    except Exception as e:
                        logger.error(f"Was unable to find neither processed datapath nor raw datapath. Make sure you provide at least one of them: {e}")
                        logger.debug(f"Congigurations found in provided config.yml path: {configurations}")
                        sys.exit()
                
                # This is used (temporarily), in the very end, for storing the dataset path used for model training: 'getattr(self, 'dataset_path', 'unknown')'
                self.dataset_path = dataset
                
                # Load dataset based on format
                self.df, self.original_df = data_utils.check_and_return_df(dataset)
                        
                # Determine target column (proper fix + simple fix)
                self.target_col = self._determine_target_column()
            
            # Not Hypothesis flow: User must provide dataset and model_type
            else:
                logger.info(f"Model Training initialized without hypothesis configuration.")
                
                if dataset is None or model_type is None:
                    logger.error(f"Model Training must be initialized with path to processed dataset and model_type! You provided dataset '{dataset}' and model_type '{model_type}'.")
                    sys.exit()
                    
                self.dataset_path = dataset
                self.model_type = model_type
                self.target_col = target_col
                
                # Set walk-forward configuration for non-hypothesis case
                self.walk_forward_config = {}
                if use_walk_forward is None:
                    # Auto-detect for time series data
                    self.use_walk_forward = False  # Will be set to True after loading data if it's time series
                else:
                    self.use_walk_forward = use_walk_forward
                
                # Load dataset based on format
                self.df, self.original_df = data_utils.check_and_return_df(dataset)
                
                # Auto-detect walk-forward for time series data in non-hypothesis case
                if use_walk_forward is None:
                    self.use_walk_forward = self._is_time_series_data()
                    if self.use_walk_forward:
                        logger.info("Auto-detected time series data, enabling walk-forward validation")
            
            # Reading directly from a .csv-file means df will have no index.
            if (self.df.index.name != 'date'):
                # DataFrame has no index (name 'date'), so setting it and dropping the duplicate
                self.df.index = pd.to_datetime(self.df['date'])  # setting index, but ensure its in datetime format
                self.df = self.df.drop('date', axis=1)
                logger.debug(f"Successfully made index to {self.df.index.name}")
            
        except Exception as e:
            logger.error(f"Error initializing Model Training: {str(e)}")
            raise
    
    def run(self):
        """
        Train initialized model type. Successfully training the model will store in according to
        the config.py-file.
        
        Returns:
            - If successful, will return the path of stored model, else False.
        """
        logger.info(f"---------- Starting to run training of {self.model_type} Model ... ----------")
        
        self.model_module, self.module_path = self._get_implemented_models()
        
        # Using imported module
        # Current implementation: if your model files have a class like XGBoostModel, DecisionTreeModel, etc.
        try:
            # Assuming each model file has a class that follows a naming pattern
            class_name = f"{self.model_type.title().replace('_', '')}Model"
            model_class = getattr(self.model_module, class_name)
            
            # Choose training approach based on configuration
            if self.use_walk_forward:
                logger.info(f"Using Walk-Forward Validation for model training")
                self.model = self._train_with_walk_forward(model_class)
            else:
                logger.info(f"Using simple temporal train/test split for model training")
                # Split data into train and test
                self.temporal_train_test_split()
                
                # Instantiate model with parameters
                model_params = getattr(self.config, 'model_training_parameters', {})
                model_instance = model_class(dataset=self.df, X_train=self.X_train, y_train=self.y_train, model_params=model_params)
                # Actually train the model
                self.training_start_time = time.time()  # not really accurate starting it here, but short term solution
                self.model = model_instance.train()
                self.model.type = self.model_type
            
            self.save_model()
            
            return self.model
            
        except AttributeError as e:
            logger.error(f"Could not find expected class in {self.module_path}: {str(e)}")
            return False
    
    def _get_processed_path(self, raw_path: str):
        """
        Convert a raw file path to its processed equivalent.
        Currently, the approach from raw to processed data (in terms of naming standards) is that
        I use the exact same name for processed as raw, but switch out the first name 'raw' with 'processed'
        (same with dir). Meaning we can use this same approach here.
        
        Args:
            raw_path (str): Path to the raw file
            
        Returns:
            str: Path to the processed file
        """
        # Convert raw path to processed path using new symbol-based structure
        processed_path = raw_path.replace("/raw/", "/processed/")
        return processed_path
    
    def _determine_target_column(self):
        """
        Intelligently determine the target column for training.
        
        Priority order:
        1. Use configured target_column if provided in hypothesis config
        2. For classification models, prefer binary targets over multi-class
        3. Fall back to first target_* column found
        
        Returns:
            str: Target column name
        """
        # Priority 1: Use configured target if provided (proper fix)
        if hasattr(self, 'configured_target') and self.configured_target:
            if self.configured_target in self.df.columns:
                logger.info(f"Using configured target column: '{self.configured_target}'")
                return self.configured_target
            else:
                logger.warning(f"Configured target '{self.configured_target}' not found in data columns")
        
        # Priority 2: Smart automatic detection (simple fix)
        target_columns = [col for col in self.df.columns if col.startswith('target_')]
        
        if not target_columns:
            logger.error("No target columns found with 'target_' prefix")
            raise ValueError("No target columns found")
        
        logger.debug(f"Found target columns: {target_columns}")
        
        # For classification models, prefer directional targets over binary targets
        if self.model_type in ['random_forest', 'logistic_regression', 'svm']:
            # Priority 1: Directional targets (three-class) for better signal quality
            directional_targets = [col for col in target_columns if 'directional' in col]
            if directional_targets:
                chosen_target = directional_targets[0]
                logger.info(f"Selected directional target for classification: '{chosen_target}'")
                return chosen_target
                
            # Priority 2: Binary targets (fallback for legacy compatibility)
            binary_targets = [col for col in target_columns if 'binary' in col]
            if binary_targets:
                chosen_target = binary_targets[0]
                logger.info(f"Selected binary target for classification: '{chosen_target}'")
                return chosen_target
        
        # For regression models, prefer continuous targets
        elif self.model_type in ['xgboost', 'linear_regression']:
            continuous_targets = [col for col in target_columns if 'binary' not in col and 'multi' not in col]
            if continuous_targets:
                chosen_target = continuous_targets[0]
                logger.info(f"Selected continuous target for regression: '{chosen_target}'")
                return chosen_target
        
        # Fall back to first target found
        chosen_target = target_columns[0]
        logger.info(f"Using first available target column: '{chosen_target}'")
        return chosen_target
        
    def _get_implemented_models(self):
        """
        Method to get implemented models by reading files in model_factory/ and matching
        it with provided model_type.
        
        If it finds a match and successfully imports the module (model file), returns the
        model_module and module_path.
        Unsuccessfull will exit.
        """
        # Find file matching provided model_type
        model_factory_dir = os.path.join(config.BASE_DIR, 'model', 'model_factory')
        model_implementation_list = os.listdir(model_factory_dir)
        model_implementation_list.remove('__pycache__') if '__pycache__' in model_implementation_list else None # No need to list this as it's not a model type
        model_implementation_list.remove('base_model.py')
        
        found_model = False  # var for wether we find the model file or not
        model_module = None  # var for potential model import
        
        # Loop through model list and find model file based on provided model type
        for model_file in model_implementation_list:
            if self.model_type in model_file and model_file.endswith('.py'):
                found_model = True
                
                # Dynamically import the module
                module_name = model_file[:-3]  # Remove .py extension
                module_path = f"model.model_factory.{module_name}"
                
                try:
                    model_module = importlib.import_module(module_path)
                    logger.debug(f"Successfully imported {module_path}")
                    return model_module, module_path
                except ImportError as e:
                    logger.error(f"Failed to import {module_path}: {str(e)}")
                    sys.exit()
        
        if not found_model or model_module is None:
            logger.error("Model Trainer could not find an implementation of wanted model type. " +
                          f"You asked for '{self.model_type}', but these are the current " +
                          f"implementations: {model_implementation_list}.")
            logger.debug("^ Remember 'model_type' must match exact name of file, eg. 'model_type == model_type.py'.")
            sys.exit()
    
    def temporal_train_test_split(self, return_datasets:bool = False):
        """
        'Complete' way of creating X and y, and getting their respecitve *_train and *_test for model training and testing.
        Train test split is done using sktime.split.temporal_train_test_split() (meaning not random as it's important for 
        avoiding lookahead bias).
        
        Args:
            - retur: In case you want to return the train and test sets for outside use. True will return them.
        """
        # Initialize train/test targets
        try:
            # Need to change potential columns having dtype object to category, for model to 'accept'
            self.df = model_utils.handle_categories(df = self.df)
            
            # Features (Input variables) - Exclude ALL target columns and future-looking features to prevent data leakage
            target_columns = [col for col in self.df.columns if col.startswith('target_')]
            future_columns = [col for col in self.df.columns if col.startswith('future_')]
            
            # NEW: Dynamic OHLC exclusion system - exclude basic OHLC features while keeping enhanced features
            # Keep only close_og (original close) and exclude normalized OHLC that may dominate hybrid features
            ohlc_columns = []
            if self.exclude_basic_ohlc:
                basic_ohlc = ['open', 'high', 'low', 'close', 'volume']  # Exclude basic OHLC
                available_ohlc = [col for col in basic_ohlc if col in self.df.columns]
                ohlc_columns = available_ohlc
                logger.info(f"DYNAMIC EXCLUSION: Excluding basic OHLC features to focus on hybrid features: {available_ohlc}")
            
            exclude_columns = target_columns + future_columns + ohlc_columns
            
            logger.info(f"Excluding columns to prevent data leakage: {exclude_columns}")
            self.X = self.df.drop(exclude_columns, axis=1, errors='ignore')
            logger.debug(f"Successfully set Features (X), with {len(self.X.columns)} columns/features (excluded {len(exclude_columns)} leaked columns)")
            # Target (Output variables)
            self.y = self.df[self.target_col]
            logger.debug(f"and Target (y), '{self.target_col}', {self.y.size} values.")
        except Exception as e:
            logger.error(f"Unable to set train/test targets: {e}")
            
        model_utils.ensure_correct_xy_format(X = self.X, y = self.y)
        
        self.y_train, self.y_test, self.X_train, self.X_test = temporal_train_test_split(y=self.y, X=self.X, test_size=0.2, train_size=0.8)  # sktime's returns values in a different order than sklearn's train_test_split
        
        if return_datasets:
            return self.y_train, self.y_test, self.X_train, self.X_test
        
    def save_model(self, filename: str = None):
        """Save trained model and metadata under timestamped folder."""
        # Track training time (you'd start this timer at beginning of training)
        # training_start_time = getattr(self, 'training_start_time', time.time())
        training_duration = time.time() - self.training_start_time
        
        current_time = datetime.datetime.now().isoformat(sep='_', timespec='seconds')
        
        # Directory and folder model and info will be saved under
        directory = os.path.join(config.TRAINED_MODEL_STORAGE, current_time)
        
        # Store the model directory for later use
        self.model_directory = directory
        
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            logger.debug(f"Creating directory now.")
            os.makedirs(directory)
        
        try:
            # Save the model itself as a pickle file
            model_pkl_file = os.path.join(directory, f"{self.model_type}.pkl")
            with open(model_pkl_file, 'wb') as file:  
                pickle.dump(self.model, file)
            logger.info(f"Successfully saved trained model: {model_pkl_file}")
            
            # Save model metadata
            self._save_model_info(directory, training_duration)
            
        except Exception as e:
            logger.error(f"Unable to save trained model: {e}")
            raise

    def _save_model_info(self, directory: str, training_duration: float):
        """Save model metadata to YAML file."""
        
        # Collect system information
        def get_system_info():
            return {
                'cpu': platform.processor() or platform.machine(),
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'python_version': platform.python_version(),
                'os': f"{platform.system()} {platform.release()}"
            }
        
        # Convert any value to a YAML-safe native Python type
        def make_yaml_safe(obj):
            """Convert numpy/pandas types to native Python types for YAML serialization."""
            import numpy as np
            import pandas as pd
            
            if obj is None:
                return None
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {key: make_yaml_safe(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_yaml_safe(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif pd.isna(obj):
                return None
            else:
                return str(obj)
        
        # Get model hyperparameters if available
        def get_model_params():
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                return make_yaml_safe(params)
            elif hasattr(self.model, '__dict__'):
                # Filter out non-parameter attributes
                params = {}
                for key, value in self.model.__dict__.items():
                    if not key.startswith('_') and not callable(value):
                        params[key] = make_yaml_safe(value)
                return params
            else:
                return {}
        
        # Create metadata dictionary
        metadata = {
            'hypothesis': {
                'name': self._hypothesis()
            },
            
            'model_info': {
                'model_type': self.model_type,
                'model_class': self.model.__class__.__name__,
                'storage_path': directory,
                'created_at': datetime.datetime.now().isoformat()
            },
            
            'training': {
                'duration_seconds': round(training_duration, 2),
                'duration_formatted': str(datetime.timedelta(seconds=int(training_duration))),
                'completed_at': datetime.datetime.now().isoformat(),
                'train_samples': int(len(self.X_train)),
                'test_samples': int(len(self.X_test)),
                'validation_method': 'walk_forward' if self.use_walk_forward else 'temporal_split',
                'walk_forward_periods': getattr(self, 'walk_forward_periods', None),
                'final_training_period': getattr(self, 'final_training_period', None)
            },
            
            'dataset': {
                'dataset_path': getattr(self, 'dataset_path', 'unknown'),  # Needs 'self.dataset_path = dataset' 
                'target_column': self.target_col,
                'feature_count': int(len(self.X_train.columns)),
                'features': self.X_train.columns.tolist(),
                'total_samples': int(len(self.df)),
                'date_range': {
                    'start': str(self.df.index.min()),
                    'end': str(self.df.index.max())
                },
                'missing_values': make_yaml_safe(self.X_train.isnull().sum().sum())
            },
            
            'model_parameters': get_model_params(),
            
            'system_info': get_system_info(),
            
            'data_split': {
                'train_size': int(len(self.X_train)),
                'test_size': int(len(self.X_test)),
                'split_ratio': f"{len(self.X_train)}/{len(self.X_test)}"
            }
        }
        
        # Save to YAML file with spacing between sections
        metadata_file = os.path.join(directory, 'model_metadata.yml')
        
        # Generate YAML string
        yaml_content = yaml.dump(metadata, default_flow_style=False, sort_keys=False, indent=2)
        
        # Add blank lines between top-level sections
        sections = ['training:', 'dataset:', 'model_parameters:', 'system_info:', 'data_split:']
        for section in sections:
            yaml_content = yaml_content.replace(f'\n{section}', f'\n\n{section}')
        
        # Write formatted YAML
        with open(metadata_file, 'w') as file:
            file.write(yaml_content)
        
        logger.info(f"Successfully saved model metadata: {metadata_file}")
    
    def _is_time_series_data(self) -> bool:
        """Check if the dataset appears to be time series data."""
        if not hasattr(self, 'df') or self.df is None:
            return False
        
        # Check if index is datetime
        if hasattr(self.df.index, 'dtype') and 'datetime' in str(self.df.index.dtype):
            return True
        
        # Check if there's a datetime column
        datetime_columns = ['date', 'datetime', 'timestamp', 'time']
        if any(col in self.df.columns for col in datetime_columns):
            return True
        
        return False
    
    def _train_with_walk_forward(self, model_class):
        """Train model using walk-forward validation approach."""
        logger.info("Implementing walk-forward validation within model training")
        
        # Get walk-forward parameters with defaults
        initial_training_months = self.walk_forward_config.get('initial_training_months', 6)
        validation_period_months = self.walk_forward_config.get('validation_period_months', 1)
        step_forward_months = self.walk_forward_config.get('step_forward_months', 1)
        
        # Import walk-forward validator to use its period creation logic
        from model.walk_forward_validator import WalkForwardValidator
        
        # Create a temporary validator instance to use its period logic
        temp_validator = WalkForwardValidator(
            hypothesis_config_path=self.hypothesis_source,
            validation_config={
                'initial_training_months': initial_training_months,
                'validation_period_months': validation_period_months,
                'step_forward_months': step_forward_months,
                'min_training_samples': 10000  # Lower threshold for model training
            }
        )
        
        # Load data and create periods
        data = temp_validator.load_data()
        periods = temp_validator.create_validation_periods(data)
        
        if not periods:
            logger.warning("No walk-forward periods created, falling back to temporal split")
            self.temporal_train_test_split()
            model_params = getattr(self.config, 'model_training_parameters', {})
            model_instance = model_class(dataset=self.df, X_train=self.X_train, y_train=self.y_train, model_params=model_params)
            self.training_start_time = time.time()
            model = model_instance.train()
            model.type = self.model_type
            return model
        
        logger.info(f"Training final model using last period with {periods[-1]['training_samples']} samples")
        
        # Use the last (most recent) period for final model training
        final_period = periods[-1]
        training_data = data.loc[final_period['training_start']:final_period['training_end']].copy()
        
        # Set up training data like the regular flow
        self.df = training_data
        self.temporal_train_test_split()
        
        # Train the final model
        model_instance = model_class(dataset=self.df, X_train=self.X_train, y_train=self.y_train)
        self.training_start_time = time.time()
        model = model_instance.train()
        model.type = self.model_type
        
        # Store walk-forward validation metadata
        self.walk_forward_periods = len(periods)
        self.final_training_period = {
            'start': final_period['training_start'].isoformat(),
            'end': final_period['training_end'].isoformat(),
            'samples': final_period['training_samples']
        }
        
        logger.info(f"Walk-forward training completed using {len(periods)} periods")
        return model

    def _hypothesis(self):
        """Simple method to check if this model is trained on a hypothesis or not, and returns json ready answer."""
        # If model is trained on hypothesis source: return json of the hypothesis_name
        if self.hypothesis_source is not None:
            return f"{config_loader.load_hypothesis_config(self.hypothesis_source).handle_hypothesis_source().get('hypothesis_name')}"
        else:  # If not, return none
            return f'{None}'