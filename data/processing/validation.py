from data.processing.base_processor import BaseProcessor
from utils import data_utils

class DataValidator(BaseProcessor):
    def __init__(self, clean_dataset):
        """
        Class for validating market data. Preferably after the data has been cleaned.
        
        Args:
            clean_dataset: path to the raw dataset (csv format). Can be either path to csv or already loaded DataFrame.
        """
        
        # Load dataset based on format
        self.df, self.original_df = data_utils.check_and_return_df(clean_dataset)
        
    def run(self):
        """Run data validation checks and returns validated results."""
        self._validate_ohlc_relationships()
        self._validate_timestamps()
        self._validate_volume()
        self._validate_price_movement()
        self._validate_data_completeness()
    
    def _validate_ohlc_relationships(self):
        """..."""
    
    def _validate_timestamps(self):
        """..."""
        
    def _validate_volume(self):
        """..."""
        
    def _validate_price_movement(self):
        """..."""
        
    def _validate_data_completeness(self):
        """..."""
    