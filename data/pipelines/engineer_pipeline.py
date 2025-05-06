import logging
import os
import pandas as pd

from utils import data_utils
from utils import shared_utils
import config.config as config
from data.processing.cleaning import DataCleaner
from data.processing.validation import DataValidator

class EngineeringPipeline():
    def __init__(self, raw_dataset : str, output_path : str = None):
        """
        Pipeline and orchestrator for processing and engineering raw datasets
        into data ready to be used for training.
        Currently only a very simple way of saving processed file: adding a _processed at the end of original name
        
        Args:
            raw_dataset: path to the raw dataset (csv format).
        """
        self.output_path = output_path

        # Loading data
        self.df, self.original_df = data_utils.check_and_return_df(raw_dataset)
        logging.debug("Engineering Pipeline - Successfully loaded data!")
        
        # Initializing processors
        self.cleaner = DataCleaner(raw_dataset=self.df)
    
    def run(self):
        """
        Run the pipeline.
        
        returns:
            pd.DataFrame: Engineered data ready to use for training.
        """
        
        # Clean the data
        self.df = self.cleaner.run()
        
        # Validate
        self.validator = DataValidator(self.df)
        self.validator_results = self.validator.run()
        self.df = self.validator_results["validated_data"]
        self.is_valid = self.validator_results["is_valid"]
        self.issues = self.validator_results["issues"]
        logging.info(f"Validation is_valid: {self.is_valid}. Validation issues: {self.issues}")
        
        # Feature Generation
        
        # Normalization
        
        # Feature Selection
        
        # Save Engineered data
        shared_utils.ensure_path_exists(path = self.output_path or config.CAPCOM_PROCESSED_DATA_DIR)
        self.df.to_csv(os.path.join(config.CAPCOM_PROCESSED_DATA_DIR, 'testing.csv'), index=True)  # since df.column["date"] is the index, we include it in the saving