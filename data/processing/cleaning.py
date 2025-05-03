import logging
import pandas as pd
import numpy as np

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
        self.df, self.original_df = data_utils.check_and_return_df(raw_dataset)
        
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
        self.df.columns = self.df.columns.str.lower()
        
        self._handle_missing_values()
        
        # ...
        
        return self.df
        
    def _handle_missing_values(self):
        """
        ...
        """
        
        # Make sure all none is the same "type"
        self.df.replace(['', 'NA', 'N/A', 'null', 'Null', 'NULL', 'None', 'NaN'], np.nan, inplace=True)
        
        # Open column
        self.df['open'] = self.df['open'].combine_first(self.df['close'].shift(1))  # Prefer previous close
        self.df['open'] = self.df['open'].interpolate()  # Fill remaining gaps

        
        # High: max(open, low)
        
        # Low: min(open.low)
        
    def _remove_duplicates(self):
        """Removes duplicates."""
        
    def _handle_outliers(self):
        """
        ...
        """
        
    def _timestamp_alignment(self):
        """
        Ensures uniform and continuous time intervals (especially in minute/hour data),
        and normalizes to a single timezone.
        """
    
    def _datatype_consistency(self):
        """Ensure correct formats: timestamps as datetime, prices as floats, volumes as integers."""
    
    def _ohlc_validity(self):
        """
        ...
        """