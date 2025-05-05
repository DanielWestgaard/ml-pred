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
        self._remove_duplicates()
        self._timestamp_alignment()
        self._handle_outliers()
        self._datatype_consistency()
        self._ohlc_validity()
        
        return self.df
        
    def _handle_missing_values(self):
        """
        Handling missing values appropriately.
        
        Current use:
            | Column | Recommended Fill                                   | Notes                             |
            | ------ | -------------------------------------------------- | --------------------------------- |
            | Open   | Forward fill / interpolate                         | Maintain continuity               |
            | High   | Max(Open, Close) or interpolate                    | Avoid inflating volatility        |
            | Low    | Min(Open, Close) or interpolate                    | Same reason                       |
            | Close  | Forward fill / interpolate                         | Most important for many models    |
            | Volume | 0 (if no trades), or interpolate if data feed lost | Never forward fill volume blindly |
        """
        
        # Make sure all none is the same "type"
        self.df.replace(['', 'NA', 'N/A', 'null', 'Null', 'NULL', 'None', 'NaN'], np.nan, inplace=True)
        
        # --- Fill CLOSE ---
        self.df['close'] = self.df['close'].ffill()  # Step 1: forward-fill
        self.df['close'] = self.df['close'].interpolate()  # Step 2: fill remaining gaps if any

        # --- Fill OPEN ---
        self.df['open'] = self.df['open'].combine_first(self.df['close'].shift(1))  # Use previous close
        self.df['open'] = self.df['open'].interpolate()  # Fill any remaining gaps
        
        # --- Fill HIGH ---
        self.df['high'] = self.df['high'].combine_first(self.df[['open', 'close']].max(axis=1))
        self.df['high'] = self.df['high'].interpolate()

        # --- Fill LOW ---
        self.df['low'] = self.df['low'].combine_first(self.df[['open', 'close']].min(axis=1))
        self.df['low'] = self.df['low'].interpolate()
        
    def _remove_duplicates(self):
        """Removes duplicates."""
        self.df = self.df.drop_duplicates(subset=['date'])
        
    def _timestamp_alignment(self):
        """
        Ensures uniform and continuous time intervals and normalizes to a single timezone.
        
        Handles:
        1. Converting timestamp column to datetime
        2. Setting datetime as index
        3. Timezone normalization to UTC
        4. Identifying gaps in time series
        """
        # Check if 'date' column exists
        if 'date' not in self.df.columns:
            logging.error("No 'date' column found in the dataframe")
            return
            
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            try:
                self.df['date'] = pd.to_datetime(self.df['date'])
            except Exception as e:
                logging.error(f"Failed to convert date column to datetime: {e}")
                return
        
        # Sort by date
        self.df = self.df.sort_values('date')        
        
        # Check for timezone info in the date column
        sample_date = self.df['date'].iloc[0]
        has_tz = hasattr(sample_date, 'tzinfo') and sample_date.tzinfo is not None
        
        # If timezone info exists but is not UTC, convert to UTC
        if has_tz and str(sample_date.tzinfo) != 'UTC':
            try:
                self.df['date'] = self.df['date'].dt.tz_convert('UTC')
                logging.debug("Converted timestamps to UTC")
            except Exception as e:
                logging.warning(f"Could not convert timezone to UTC: {e}")
        
        # If no timezone info exists, assume UTC
        elif not has_tz:
            try:
                self.df['date'] = self.df['date'].dt.tz_localize('UTC')
                logging.debug("Localized timestamps to UTC")
            except Exception as e:
                logging.warning(f"Could not localize timezone to UTC: {e}")

        # Set date as index if it's not already
        if self.df.index.name != 'date':
            self.df = self.df.set_index('date')
        
        # Identify frequency of data (daily, hourly, minute, etc.)
        if len(self.df) > 1:
            time_diffs = self.df.index.to_series().diff().dropna()
            common_diff = time_diffs.mode().iloc[0]
            logging.debug(f"Most common time interval: {common_diff}")
            
            # Check for gaps
            gaps = time_diffs[time_diffs > common_diff * 1.5]
            if not gaps.empty:
                logging.warning(f"Found {len(gaps)} gaps in time series")
                # Optionally fill gaps with NaN and then forward fill
                # full_idx = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq=common_diff)
                # self.df = self.df.reindex(full_idx)
                # Then you'd need to handle the NaNs appropriately
        
    def _handle_outliers(self):
        """
        Detect and handle outliers in OHLCV data using multiple methods.
        
        Implements:
        1. IQR method for price columns
        2. Z-score method for volume
        3. Winsorization for extreme values
        4. OHLC relationship validation
        """
        # Define price columns
        price_cols = ['open', 'high', 'low', 'close']
        
        # --- METHOD 1: IQR for price columns ---
        for col in price_cols:
            # Calculate IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Flag potential outliers (for logging)
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            if len(outliers) > 0:
                logging.debug(f"Found {len(outliers)} outliers in {col} using IQR method")
            
            # Winsorize (cap) extreme values instead of removing
            self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # --- METHOD 2: Z-score for volume (more robust for highly skewed data) ---
        if 'volume' in self.df.columns:
            # Log transform first (volume is usually right-skewed)
            log_volume = np.log1p(self.df['volume'])  # log(1+x) to handle zeros
            
            # Calculate z-score
            mean_log_vol = log_volume.mean()
            std_log_vol = log_volume.std()
            z_scores = np.abs((log_volume - mean_log_vol) / std_log_vol)
            
            # Flag potential outliers (z-score > 3)
            volume_outliers = self.df[z_scores > 3]
            if len(volume_outliers) > 0:
                logging.debug(f"Found {len(volume_outliers)} volume outliers using Z-score method")
            
            # Cap extreme values (in log space, then transform back)
            cap_log_vol = log_volume.clip(upper=mean_log_vol + 3*std_log_vol)
            self.df['volume'] = np.expm1(cap_log_vol)  # exp(x)-1 to reverse log1p
    
    def _datatype_consistency(self):
        """Ensure correct formats: timestamps as datetime, prices as floats, volumes as integers."""
        logging.warning("Datatype consistency not implemented yet.")
    
    def _ohlc_validity(self):
        """
        ...
        """
        logging.warning("ohlc validity not implemented yet.")