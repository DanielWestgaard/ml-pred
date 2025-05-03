import logging
import pandas as pd

from data.processing.cleaning import DataCleaner
from utils import data_utils


class EngineeringPipeline():
    def __init__(self, raw_dataset : str):
        """
        Pipeline and orchestrator for processing and engineering raw datasets
        into data ready to be used for training.
        
        Args:
            raw_dataset: path to the raw dataset (csv format).
        """

        # Loading data
        self.data, self.original_data = data_utils.check_and_return_df(raw_dataset)
        logging.debug("Engineering Pipeline - Successfully loaded data!")
        
        # Initializing processors
        self.cleaner = DataCleaner(raw_dataset=self.data)
        
    
    def run(self):
        """
        Run the pipeline.
        
        returns:
            pd.DataFrame: Engineered data ready to use for training.
        """
        
        # Clean the data
        self.cleaner.run()
        
        # Validate 
        
        # Feature Generation
        
        # Feature Selection
        
        # Save Engineered data
        