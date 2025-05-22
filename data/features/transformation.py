import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from data.processing.base_processor import BaseProcessor
from utils import data_utils


class FeatureTransformer(BaseProcessor):
    def __init__(self, data):
        """Class for transforming- (handling nan's, encoding categorical variables, etc.) and normalizing features."""
        # Load dataset based on format
        self.df, self.original_df = data_utils.check_and_return_df(data)
    
    def run(self, preserve_original=False, window=20):
        """
        Run feature transformation.
        
        Args:
            preserve_original: If True, keep original features and add normalized ones with suffix
            window: Size of rolling window for normalization statistics
        """
        # Handle missing feature values
        self.handle_missing_values()
        
        # Handle duplicates
        self.handle_duplicates()
        
        # Normalization
        self.normalize(preserve_original=preserve_original, window=window)
        
        return self.df
    
    def handle_missing_values(self):
        """
        Method for handling missing values in features:
        1. Drop columns with excessive missing values (threshold-based)
        2. Drop initial rows based on the max window size (warmup period)
        3. Apply forward fill to handle any remaining scattered NaN values
        """
        # Check if the dataset has any missing feature
        if self.df.isnull().values.any():
            logging.warning("Dataset has features containing empty values. Will drop relevant features.")

            self._filter_features()

            self._handle_scattered_nans()
        else:
            logging.debug("Dataset does not contain features with missing values.")
    
    def _filter_features(self, max_window_size:int = None, threshold:int = 0.5):
        """
        Filter features based on missing values.
        
        Parameters:
            - max_window_size: The feature with highest window size. Meaning minimum number of values needed to calculate.
            - threshold: Percentage of how many empty values a feature can have before being removed.
        """
        logging.debug("Starting to filter features based on missing values...")
        
        nan_percentage = self.df.isna().mean()  # Calculate the percentage of NaN values in each column
        # Firstly dropping features with many missing values
        self.df = self.df.drop(columns=nan_percentage[nan_percentage > threshold].index)
        removed = self.original_df.columns.difference(self.df.columns)
        logging.debug(f"Dropped {len(removed)} columns that exceeded threshold for NaN's: {removed}")

        # Secondly, drop first rows based on the highest window size
        if max_window_size is None:
            high_feature_windows = ["sma_200", "sma_50", "sma_20", "sma_10", "sma_5"]  # highest possible (feature-) window must be first
            for feature in high_feature_windows:
                if True in self.df.columns.str.contains(feature):
                    logging.debug(f"Found {feature} first in df.")
                    # This approach assumes all features have feature name in "one word"/abbreviation, followed by '_' and the window size
                    bla = feature.split('_')
                    try:
                        if len(self.df) > int(bla[1]):
                            max_window_size = int(bla[1])
                            logging.debug(f"Successfully extracted highest window size from feature list: {max_window_size}.")
                            break
                        else:
                            logging.debug(f"The matched feature {feature} have a longer window, {int(bla[1])}, than the length of the df, {len(self.df)}. Continuing...")                            
                    except Exception as e: 
                        logging.error(f"Failed getting window size from feature name in column or converting it to int: {e}")
        # Actually dropping the first max_window_size of df
        self.df = self.df.iloc[max_window_size:]
        logging.info(f"Data Transformation - _filter_features - Successfully dropped first {max_window_size} rows. Size of df is now {len(self.df)}")
        
    def _handle_scattered_nans(self):
        """
        Handle any remaining scattered NaN values using forward fill,
        which is the preferred method for financial time series as it
        does not introduce lookahead bias.
        """
        # Check which columns still have NaNs before imputation
        cols_with_nans = {col: self.df[col].isnull().sum() for col in self.df.columns 
                        if self.df[col].isnull().any()}
        
        if cols_with_nans:
            logging.info(f"Applying forward fill to {len(cols_with_nans)} columns with scattered NaNs")
            
            # Store the number of NaNs before imputation
            total_nans_before = self.df.isnull().sum().sum()
            
            # Apply forward fill
            self.df = self.df.fillna(method='ffill')
            
            # Count remaining NaNs (if any)
            remaining_nans = self.df.isnull().sum().sum()
            filled_nans = total_nans_before - remaining_nans
            
            logging.info(f"Forward fill imputed {filled_nans} values")
            
            # If there are still NaNs at the beginning of the series (where ffill can't work)
            if remaining_nans > 0:
                logging.warning(f"After forward fill, {remaining_nans} NaN values remain")
                logging.warning("These are likely at the beginning of the series - consider additional treatment")
                
                # Optional: List columns that still have NaNs
                cols_still_with_nans = [col for col in self.df.columns if self.df[col].isnull().any()]
                if cols_still_with_nans:
                    logging.debug(f"Columns still containing NaNs: {cols_still_with_nans}")
                    
                    # Option 1: Fill beginning NaNs with first valid value (backward fill limited to start)
                    # This is a common approach for the beginning of a time series
                    for col in cols_still_with_nans:
                        # Find first valid index
                        first_valid_idx = self.df[col].first_valid_index()
                        if first_valid_idx is not None:
                            first_valid_value = self.df.loc[first_valid_idx, col]
                            # Fill NaNs before this index with the first valid value
                            self.df.loc[:first_valid_idx, col] = self.df.loc[:first_valid_idx, col].fillna(first_valid_value)
                    
                    logging.info("Filled beginning NaNs with first valid values")
    
    def handle_duplicates(self):
        """..."""
        # Check for duplicate columns and fix them
        duplicate_cols = self.df.columns[self.df.columns.duplicated()].tolist()
        if duplicate_cols:
            logging.warning(f"Found duplicate columns: {duplicate_cols}")
            # Rename duplicates with a suffix
            for col in duplicate_cols:
                # Get all occurrences of the duplicated column
                cols = self.df.columns.get_indexer_for([col])
                # Rename all but the first occurrence
                for i, idx in enumerate(cols[1:], 1):
                    new_name = f"{col}_{i}"
                    self.df.columns.values[idx] = new_name
                    logging.debug(f"Renamed duplicate column '{col}' to '{new_name}'")
    
    def normalize(self, preserve_original=False, window=20):
        """Perform normalization"""
        # First identify categorical columns to exclude from normalization
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Also identify binary/regime columns (that might be numeric but shouldn't be normalized)
        regime_indicators = [col for col in self.df.columns if 
                            any(x in col for x in ['regime', 'is_', 'in_value_area', 'state'])]
        exclude_from_normalization = categorical_cols + regime_indicators
        
        # Group 1: Price data (OHLC) - Z-score normalize
        price_cols = ['open', 'high', 'low', 'close']
        price_cols_present = [col for col in price_cols if col in self.df.columns 
                            and col not in exclude_from_normalization]
        
        # Group 2: Volume data - Log transform then Z-score
        volume_cols = ['volume'] + [col for col in self.df.columns 
                                if 'volume' in col.lower() and 'relative' not in col.lower()]
        volume_cols = [col for col in volume_cols if col in self.df.columns 
                    and col not in exclude_from_normalization]
        
        # Group 3: Unbounded features - Z-score
        z_score_features = [col for col in self.df.columns if 
                        any(x in col for x in ['roc_', 'macd', 'vwap', 'log_return', 'distance', 'vol_', 
                                            'atr', 'bb_width', 'obv', 'cci', 'adx'])]
        z_score_features = [col for col in z_score_features if col not in exclude_from_normalization]
        
        # Group 4: Bounded indicators - Min-Max
        minmax_features = [col for col in self.df.columns if 
                        any(x in col for x in ['rsi', 'stoch', 'mfi', '_sin_', '_cos_'])]
        minmax_features = [col for col in minmax_features if col not in exclude_from_normalization]

        # Log transform volume first
        volume_cols_for_zscore = []
        for col in volume_cols:
            if preserve_original:
                log_col_name = f"{col}_log"
                self.df[log_col_name] = np.log1p(self.df[col])  # log(1+x) to handle zeros
                volume_cols_for_zscore.append(log_col_name)  # Use log-transformed column for z-score
            else:
                self.df[col] = np.log1p(self.df[col])
                volume_cols_for_zscore.append(col)  # Use the same column name
        
        # Apply z-score to price, transformed volume, and other unbounded features
        all_zscore_cols = price_cols_present + volume_cols_for_zscore + z_score_features
        self.z_score(columns=all_zscore_cols, window=window, 
                    preserve_original=preserve_original)
        
        # Apply min-max to bounded features using rolling window
        if minmax_features:
            self.rolling_minmax(columns=minmax_features, window=window,
                            preserve_original=preserve_original)
    
    def z_score(self, columns, window=20, preserve_original=False):
        """
        Z-score normalize selected features using rolling window.
        
        Parameters:
        -----------
        columns : list
            List of columns to normalize
        window : int
            The rolling window to use for calculating mean and standard deviation
        preserve_original : bool
            If True, keeps original columns and adds new ones with _norm suffix
        """
        for col in columns:
            if col not in self.df.columns:
                continue
            
            # Calculate rolling mean and standard deviation
            rolling_mean = self.df[col].rolling(window=window).mean()
            rolling_std = self.df[col].rolling(window=window).std()
            
            # Calculate z-score with division by zero handling
            z_score = pd.Series(float('nan'), index=self.df.index)
            mask = rolling_std != 0
            
            # Only calculate where std is not zero
            z_score[mask] = (self.df[col][mask] - rolling_mean[mask]) / rolling_std[mask]
            
            # For cases where std is zero, set z-score to 0
            z_score[~mask & ~rolling_mean.isna()] = 0
            
            # Forward fill NaN values from initial window
            z_score = z_score.bfill()  # z_score.fillna(method='bfill')
            
            # Add to dataframe
            if preserve_original:
                self.df[f'{col}_norm'] = z_score
            else:
                if col == 'close':  # preserve original closing value (for placing trades)
                    self.df['close_og'] = self.df['close']
                self.df[col] = z_score
    
    def rolling_minmax(self, columns, window=20, preserve_original=False):
        """
        Apply min-max scaling using rolling window to avoid lookahead bias.
        
        Parameters:
        -----------
        columns : list
            List of columns to normalize
        window : int
            The rolling window to use for calculating min and max
        preserve_original : bool
            If True, keeps original columns and adds new ones with _norm suffix
        """
        for col in columns:
            if col not in self.df.columns:
                continue
                
            # Calculate rolling min and max
            rolling_min = self.df[col].rolling(window=window).min()
            rolling_max = self.df[col].rolling(window=window).max()
            
            # Calculate min-max scaled values with division by zero handling
            scaled = pd.Series(float('nan'), index=self.df.index)
            mask = (rolling_max - rolling_min) != 0
            
            # Only calculate where range is not zero
            scaled[mask] = (self.df[col][mask] - rolling_min[mask]) / (rolling_max[mask] - rolling_min[mask])
            
            # For cases where range is zero, set to 0.5 (middle of range)
            scaled[~mask & ~rolling_min.isna()] = 0.5
            
            # Forward fill NaN values from initial window
            scaled = scaled.bfill()  # scaled.fillna(method='bfill')
            
            # Add to dataframe
            if preserve_original:
                self.df[f'{col}_norm'] = scaled
            else:
                self.df[col] = scaled