import logging
import pandas as pd

from data.processing.base_processor import BaseProcessor
import utils.data_utils as data_utils


class DataCleaner(BaseProcessor):
    def __init__(self, raw_dataset):
        """
        Class for cleaning market data.
        
        Args:
            raw_dataset: path to the raw dataset (csv format). Can be either path to csv or already loaded DataFrame.
        """
        
        # Load dataset based on format
        self.data = data_utils.check_and_return_df(raw_dataset)
        
    def run(self):
        """
        Clean the data.
        
        Currently covers handling missing data, removing duplicates, outlier handling, timestamp alignement,
        datatype consistency, and OHLC validity.
        This is non-exhaustive, so I will (hopefully) add and improve these steps later.
        TODO: 
            - Corporate Action Adjustments: Adjust for splits and dividends if working with raw price data, 
              or perhaps using adjusted close prices, or do your own adjustments.
        
        returns:
            pd.DataFrame: Cleaned dataset.
        """
        # Standardizing column names to lower case
        self.data.columns = self.data.columns.str.lower()
        
    def _handle_missing_values(self):
        """
        ...
        """
        
    def _remove_duplicates(data : pd.DataFrame):
        """Removes duplicates."""
        
    def _handle_outliers(data : pd.DataFrame):
        """
        ...
        """
        
    def _timestamp_alignment(data : pd.DataFrame):
        """
        Ensures uniform and continuous time intervals (especially in minute/hour data),
        and normalizes to a single timezone.
        """
    
    def _datatype_consistency(data : pd.DataFrame):
        """Ensure correct formats: timestamps as datetime, prices as floats, volumes as integers."""
    
    def _ohlc_validity(data : pd.DataFrame):
        """
        ...
        """