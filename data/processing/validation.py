from data.processing.base_processor import BaseProcessor
from utils import data_utils

class DataValidation(BaseProcessor):
    def __init__(self, clean_dataset):
        """
        Class for validating market data. Preferably after the data has been cleaned.
        
        Args:
            clean_dataset: path to the raw dataset (csv format). Can be either path to csv or already loaded DataFrame.
        """
        
        # Load dataset based on format
        self.df, self.original_df = data_utils.check_and_return_df(clean_dataset)
        
    def run(self):
        """Run data validation for ohlcv date market data."""
        pass
    
    def _check_boundries(self):
        """..."""
        
    def _check_boundries(self):
        """..."""
    