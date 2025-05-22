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
        if 'close' in self.df.columns:
            self.df['close'] = self.df['close'].ffill()  # Step 1: forward-fill
            self.df['close'] = self.df['close'].interpolate()  # Step 2: fill remaining gaps if any

        # --- Fill OPEN ---
        if 'open' in self.df.columns:
            if 'close' in self.df.columns:
                self.df['open'] = self.df['open'].combine_first(self.df['close'].shift(1))  # Use previous close
            self.df['open'] = self.df['open'].interpolate()  # Fill any remaining gaps
        
        # --- Fill HIGH ---
        if 'high' in self.df.columns:
            # Only use available columns for the fallback calculation
            available_price_cols = [col for col in ['open', 'close'] if col in self.df.columns]
            if available_price_cols:
                self.df['high'] = self.df['high'].combine_first(self.df[available_price_cols].max(axis=1))
            self.df['high'] = self.df['high'].interpolate()

        # --- Fill LOW ---
        if 'low' in self.df.columns:
            # Only use available columns for the fallback calculation
            available_price_cols = [col for col in ['open', 'close'] if col in self.df.columns]
            if available_price_cols:
                self.df['low'] = self.df['low'].combine_first(self.df[available_price_cols].min(axis=1))
            self.df['low'] = self.df['low'].interpolate()
        
        # --- Fill VOLUME ---
        if 'volume' in self.df.columns:
            # Step 1: If price didn't change (high=low), it's likely no trades occurred
            # Only check this if both high and low columns exist
            if 'high' in self.df.columns and 'low' in self.df.columns:
                no_movement_mask = (self.df['high'] == self.df['low'])
                self.df.loc[no_movement_mask & self.df['volume'].isna(), 'volume'] = 0
            
            # Step 2: For other missing values, use interpolation rather than forward fill
            # Linear interpolation works well for shorter gaps
            self.df['volume'] = self.df['volume'].interpolate(method='linear')
            # Step 3: Any remaining NaNs at the beginning can be filled with early known volumes
            # (avoid forward fill for long sequences)
            self.df['volume'] = self.df['volume'].bfill().fillna(0)
        
    def _remove_duplicates(self):
        """Removes duplicates."""
        # Check if 'date' column exists for subset-based deduplication
        if 'date' in self.df.columns:
            self.df = self.df.drop_duplicates(subset=['date'])
        else:
            # If no date column, just remove completely identical rows
            logging.warning("No 'date' column found for deduplication, removing completely identical rows")
            self.df = self.df.drop_duplicates()
        
    def _timestamp_alignment(self):
        """
        Ensures uniform and continuous time intervals and normalizes to a single timezone.
        
        Handles:
        1. Converting timestamp column to datetime
        2. Setting datetime as index
        3. Timezone normalization to UTC
        4. Identifying and FILLING gaps in time series  # ← IMPROVED
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
        
        # Handle timezone (existing code)
        sample_date = self.df['date'].iloc[0]
        has_tz = hasattr(sample_date, 'tzinfo') and sample_date.tzinfo is not None
        
        if has_tz and str(sample_date.tzinfo) != 'UTC':
            try:
                self.df['date'] = self.df['date'].dt.tz_convert('UTC')
                logging.debug("Converted timestamps to UTC")
            except Exception as e:
                logging.warning(f"Could not convert timezone to UTC: {e}")
        elif not has_tz:
            try:
                self.df['date'] = self.df['date'].dt.tz_localize('UTC')
                logging.debug("Localized timestamps to UTC")
            except Exception as e:
                logging.warning(f"Could not localize timezone to UTC: {e}")

        # Set date as index if it's not already
        if self.df.index.name != 'date':
            self.df = self.df.set_index('date')
        
        # FIXED: Remove any remaining duplicates from the index before reindexing
        if self.df.index.duplicated().any():
            logging.warning("Found duplicate index values after setting date as index - removing")
            self.df = self.df[~self.df.index.duplicated(keep='first')]
        
        # NEW: Fill time series gaps
        if len(self.df) > 1:
            time_diffs = self.df.index.to_series().diff().dropna()
            if len(time_diffs) > 0:  # Check if we have any time differences
                common_diff = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else time_diffs.median()
                logging.debug(f"Most common time interval: {common_diff}")
                
                # Check for gaps
                gaps = time_diffs[time_diffs > common_diff * 1.5]
                if not gaps.empty:
                    logging.warning(f"Found {len(gaps)} gaps in time series - filling them")
                    
                    # Create complete time index
                    full_idx = pd.date_range(
                        start=self.df.index.min(), 
                        end=self.df.index.max(), 
                        freq=common_diff
                    )
                    
                    # Reindex to fill gaps
                    self.df = self.df.reindex(full_idx)
                    
                    # Fill the gaps using forward fill for OHLC
                    price_cols = ['open', 'high', 'low', 'close']
                    for col in price_cols:
                        if col in self.df.columns:
                            self.df[col] = self.df[col].ffill()
                    
                    # For volume, use 0 for gap periods (no trading)
                    if 'volume' in self.df.columns:
                        self.df['volume'] = self.df['volume'].fillna(0)
                    
                    logging.info(f"Filled time series gaps. New length: {len(self.df)}")

    def _handle_outliers(self):
        """
        Detect and handle outliers in OHLCV data using multiple methods.
        IMPROVED: Use consistent thresholds with validation and better volume handling
        """
        # Define price columns
        price_cols = ['open', 'high', 'low', 'close']
        
        # --- METHOD 1: IQR for price columns ---
        for col in price_cols:
            if col in self.df.columns:
                # Calculate IQR
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds (more aggressive cleaning)
                lower_bound = Q1 - 2 * IQR  # Slightly more aggressive
                upper_bound = Q3 + 2 * IQR
                
                # Flag potential outliers (for logging)
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                if len(outliers) > 0:
                    logging.debug(f"Found {len(outliers)} outliers in {col} using IQR method")
                
                # Winsorize (cap) extreme values instead of removing
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # --- METHOD 2: Enhanced volume outlier handling ---
        if 'volume' in self.df.columns:
            # First, handle negative volumes
            negative_mask = self.df['volume'] < 0
            if negative_mask.any():
                logging.debug(f"Found {negative_mask.sum()} negative volume values - setting to 0")
                self.df.loc[negative_mask, 'volume'] = 0
            
            # Filter positive volumes for outlier detection
            positive_mask = self.df['volume'] > 0
            if positive_mask.any():
                positive_volumes = self.df.loc[positive_mask, 'volume']
                
                # Method 1: IQR-based capping (more aggressive for extreme outliers)
                Q1 = positive_volumes.quantile(0.25)
                Q3 = positive_volumes.quantile(0.75)
                IQR = Q3 - Q1
                upper_bound_iqr = Q3 + 3 * IQR  # 3x IQR for volume outliers
                
                # Method 2: Log-normal based capping (for less extreme outliers)
                log_volume = np.log1p(positive_volumes)
                mean_log_vol = log_volume.mean()
                std_log_vol = log_volume.std()
                
                # Use more aggressive threshold for cleaning (2-sigma instead of 4)
                threshold = 2.5  # More aggressive than original
                upper_bound_log = mean_log_vol + threshold * std_log_vol
                upper_bound_log_val = np.expm1(upper_bound_log)
                
                # Use the more restrictive of the two bounds
                final_upper_bound = min(upper_bound_iqr, upper_bound_log_val)
                
                # Count outliers before capping
                outlier_count = (positive_volumes > final_upper_bound).sum()
                if outlier_count > 0:
                    logging.debug(f"Capping {outlier_count} extreme volume values")
                
                # Apply capping - explicitly convert to avoid dtype warning
                capped_volumes = self.df.loc[positive_mask, 'volume'].clip(upper=final_upper_bound)
                self.df.loc[positive_mask, 'volume'] = capped_volumes.astype('int64')

                # Ensure all volume values are integers
                self.df['volume'] = self.df['volume'].astype('int64')
    
    def _datatype_consistency(self):
        """Ensure correct formats: timestamps as datetime, prices as floats, volumes as integers."""
        # Ensure index is datetime if not done in timestamp_alignment
        if not isinstance(self.df.index, pd.DatetimeIndex):
            try:
                self.df.index = pd.to_datetime(self.df.index)
                logging.debug("Converted index to datetime")
            except Exception as e:
                logging.error(f"Failed to convert index to datetime: {e}")
        
        # Ensure price columns are float
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in self.df.columns:
                try:
                    original_type = self.df[col].dtype
                    self.df[col] = self.df[col].astype(float)
                    if original_type != self.df[col].dtype:
                        logging.info(f"Converted {col} from {original_type} to {self.df[col].dtype}")
                except Exception as e:
                    logging.error(f"Failed to convert {col} to float: {e}")
        
        # Ensure volume is integer
        if 'volume' in self.df.columns:
            try:
                original_type = self.df['volume'].dtype
                # First convert to float to handle any decimals, then to int
                self.df['volume'] = self.df['volume'].astype(float).round().astype('int64')
                if original_type != self.df['volume'].dtype:
                    logging.debug(f"Converted volume from {original_type} to {self.df['volume'].dtype}")
            except Exception as e:
                logging.error(f"Failed to convert volume to int: {e}")
    
    def _ohlc_validity(self):
        """
        Verify and correct OHLC logical relationships:
        
        1. High should be the highest value (≥ Open, Close, Low)
        2. Low should be the lowest value (≤ Open, Close, High)
        3. Open and Close should be between High and Low
        4. Volume should be non-negative
        
        Only processes columns that exist in the dataframe.
        """
        # Count original issues for logging
        issues_count = 0
        
        # Get available OHLC columns
        available_ohlc = [col for col in ['open', 'high', 'low', 'close'] if col in self.df.columns]
        
        # Only proceed if we have at least some OHLC columns
        if len(available_ohlc) < 2:
            logging.warning("Insufficient OHLC columns for relationship validation")
            return
        
        # Check and fix: High should be ≥ max(Open, Close) - only if all relevant columns exist
        if all(col in self.df.columns for col in ['high', 'open', 'close']):
            high_issues = self.df[self.df['high'] < self.df[['open', 'close']].max(axis=1)]
            if not high_issues.empty:
                issues_count += len(high_issues)
                logging.warning(f"Found {len(high_issues)} records where high < max(open, close) - fixing")
                self.df['high'] = self.df[['high', 'open', 'close']].max(axis=1)
        
        # Check and fix: Low should be ≤ min(Open, Close) - only if all relevant columns exist
        if all(col in self.df.columns for col in ['low', 'open', 'close']):
            low_issues = self.df[self.df['low'] > self.df[['open', 'close']].min(axis=1)]
            if not low_issues.empty:
                issues_count += len(low_issues)
                logging.warning(f"Found {len(low_issues)} records where low > min(open, close) - fixing")
                self.df['low'] = self.df[['low', 'open', 'close']].min(axis=1)
        
        # Ensure volume is non-negative - only if volume column exists
        if 'volume' in self.df.columns:
            volume_issues = self.df[self.df['volume'] < 0]
            if not volume_issues.empty:
                issues_count += len(volume_issues)
                logging.warning(f"Found {len(volume_issues)} records with negative volume - fixing")
                self.df['volume'] = self.df['volume'].clip(lower=0)
        
        # Final verification - check relationship again after fixes (only for available columns)
        remaining_issues = 0
        
        if all(col in self.df.columns for col in ['high', 'low']):
            remaining_issues += (self.df['high'] < self.df['low']).sum()
        
        if all(col in self.df.columns for col in ['high', 'open']):
            remaining_issues += (self.df['high'] < self.df['open']).sum()
        
        if all(col in self.df.columns for col in ['high', 'close']):
            remaining_issues += (self.df['high'] < self.df['close']).sum()
        
        if all(col in self.df.columns for col in ['low', 'open']):
            remaining_issues += (self.df['low'] > self.df['open']).sum()
        
        if all(col in self.df.columns for col in ['low', 'close']):
            remaining_issues += (self.df['low'] > self.df['close']).sum()
        
        if remaining_issues > 0:
            logging.warning(f"After corrections, {remaining_issues} OHLC relationship issues remain")
        elif issues_count > 0:
            logging.info(f"Successfully fixed {issues_count} OHLC relationship issues")
        else:
            logging.info("No OHLC relationship issues found!")