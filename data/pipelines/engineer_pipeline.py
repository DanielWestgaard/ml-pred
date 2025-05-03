import pandas as pd

from data.processing.cleaning import DataCleaner


class EngineeringPipeline():
    def __init__(self, raw_dataset):
        """
        Pipeline and orchestrator for processing and engineering raw datasets
        into data ready to be used for training.
        
        Args:
            raw_dataset: path to the raw dataset (csv format).
        """
        # Loading data
        self.data = pd.read_csv(raw_dataset)
        self.original_data = self.data.copy()  # Kepping the original just in case
        
        # Initializing processors
        self.cleaner = DataCleaner()
        
    
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
        