import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
        # TODO: Handle missing feature values
        self.handle_missing_values()
        
        # TODO: Encode categorical variables/values
        self.encode_categorical_vars()
        
        # Normalization
        self.normalize(preserve_original=preserve_original, window=window)
        
        return self.df
    
    def handle_missing_values(self):
        """
        Method for handling missing values in features. NaN's should be handled during feature
        generation. If there are still features with (excessive) missing values, they will be dropped.
        """
        # Check if the dataset has any missing feature
        if self.df.isnull().values.any():
            logging.warning("Dataset has features containing Null's. Will drop features.")
            # Replace blank values with DataFrame.replace() method with nan
            # self.df = self.df.replace(r'^\s*$', np.nan, regex=True)
            self._filter_features()
        else:
            logging.debug("Dataset does not contain features with missing values.")
    
    def _filter_features(self, max_window_size:int = None, threshold:int = 0.5, filter:bool = True):
        """
        Filter features based on the type and how many missing there are.
        This current 'version' is relatively simple:
            - First it drops columns containing NaN's with threshold (percentage) compared to the length
            - For features needing "warmup" before working (like sma_5, rsi_14), we'll drop the first n
            rows based on the highest window size found.
        """
        logging.debug("Starting to filter features based on missing values...")
        
        # Firstly dropping features with many null's
        if filter:
            # Calculate the percentage of NaN values in each column
            nan_percentage = self.df.isna().mean()
            # Alternatively, modify the original DataFrame
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
                        max_window_size = int(bla[1])
                        logging.debug(f"Successfully extracted highest window size from feature list: {max_window_size}.")
                        break
                    except Exception as e: 
                        logging.error(f"Failed getting window size from feature name in column or converting it to int: {e}")
        # Actually dropping the first max_window_size of df
        self.df = self.df.iloc[max_window_size:].reset_index(drop=True)
        logging.info(f"Data Transformation - _filter_features - Successfully dropped first {max_window_size} rows. Size of df is now {len(self.df)}")

        # TODO: How do we handle features that have null/nan spread across? Dropping these rows (in the middle of df) causes "wholes" which is not good.
        # Perhaps wimply dropping the feature then?
        
    
    def encode_categorical_vars(self):
        """Converting categorical/textual data into numerical format. Like One-hot encoding, label encoding."""
        pass
    
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
        for col in volume_cols:
            if preserve_original:
                self.df[f"{col}_log"] = np.log1p(self.df[col])  # log(1+x) to handle zeros
            else:
                self.df[col] = np.log1p(self.df[col])
        
        # Apply z-score to price, transformed volume, and other unbounded features
        all_zscore_cols = price_cols_present + volume_cols + z_score_features
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
            z_score = z_score.fillna(method='bfill')
            
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
            scaled = scaled.fillna(method='bfill')
            
            # Add to dataframe
            if preserve_original:
                self.df[f'{col}_norm'] = scaled
            else:
                self.df[col] = scaled