import logging
import pandas as pd
import numpy as np
import scipy
from scipy.fft import fft, fftfreq
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
import time
from contextlib import contextmanager

from utils import data_utils


class FeatureGenerator():
    def __init__(self, dataset, selected_features: list = None):
        """
        Class focusing on generating features with helpful prediction powers,
        that might help a ML model learn and predict better.
        
        Args:
            dataset: The cleaned (and validated) data, containing (index) date/datetime, and the columns OHLCV.
        
        returns:
            dataset with all generated features 
        """
        try:
            # Load dataset based on format
            self.df, self.original_df = data_utils.check_and_return_df(dataset)
            
            self.selected_features = selected_features
            
            # Validate that required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}. Dataset must contain OHLCV data.")
            
            # Check for empty dataset
            if len(self.df) == 0:
                raise ValueError("Empty dataset provided. Cannot generate features.")
                
            # Check for sufficient data (at least 200 rows for long-term features)
            if len(self.df) < 200:
                logging.warning(f"Dataset contains only {len(self.df)} rows. Some features requiring longer lookback periods may not be generated properly.")
                
            # Check for NaN values in required columns
            nan_counts = self.df[required_columns].isna().sum()
            if nan_counts.sum() > 0:
                logging.warning(f"Dataset contains NaN values: {nan_counts}. Some features may be affected.")
                
        except Exception as e:
            logging.error(f"Error initializing FeatureGenerator: {str(e)}")
            raise
        
    def run(self):
        """Run feature generation on provided dataset."""
        logging.info("---------- Running Feature Generator ----------")
        generated_features = []
                
        try:
            if self.selected_features is not None:
                logging.debug(f"Starting to generate selected features: {self.selected_features}")

                for feature_name, feature_value in self.selected_features.items():
                    try:
                        # print(f"{feature_name} : {feature_value}")
                        self.safely_execute(feature_name, feature_name)
                    except Exception as e:
                        logging.error(f"Unable to execute {feature_name}, make sure the feature name matches excactly with the one implemented: {e}")
            
            else:
                logging.debug("Starting to generate all features...")
                
                # TODO: Instead of manually calling each file, consider using shared_utils.list_of_functions_in_class()
                
                # Price Action Features
                self.safely_execute("moving_averages", "Moving Averages")
                self.safely_execute("rate_of_change", "Rate of Change")
                self.safely_execute("average_true_range", "Average True Range")
                self.safely_execute("bollinger_bands", "Bollinger Bands")
                self.safely_execute("support_resistance", "Support and Resistance")
                logging.debug("Finished calculating Price Action Features")
                
                # Volume-Based Features
                self.safely_execute("volume_moving_averages", "Volume Moving Averages")
                self.safely_execute("volume_rate_of_change", "Volume Rate of Change")
                self.safely_execute("on_balance_volume", "On Balance Volume")
                self.safely_execute("volume_profile", "Volume Profile")
                self.safely_execute("fx_activity_profile", "Activity Profile (tick volume)")  # This is only meant for Forex pairs, but maybe selection will sort this out anyways?
                self.safely_execute("vwap", "VWAP")
                self.safely_execute("money_flow_index", "Money Flow Index")
                logging.debug("Finished calculating Volume-Based Features")
                
                # Technical Indicators
                self.safely_execute("relative_strength_index", "Relative Strength Index")
                self.safely_execute("moving_average_convergence_divergence", "MACD")
                self.safely_execute("average_directional_index", "Average Directional Index")
                self.safely_execute("stochastic_oscillator", "Stochastic Oscillator")
                self.safely_execute("commodity_channel_index", "Commodity Channel Index")
                logging.debug("Finished calculating Technical Indicators")
                
                # Time-based Features
                self.safely_execute("time_of_day", "Time of Day")
                self.safely_execute("day_of_week", "Day of Week")
                self.safely_execute("time_of_week", "Time of Week")
                self.safely_execute("time_of_month", "Time of Month")
                self.safely_execute("fx_trading_sessions", "Trading Sessions")  # Also meant for Forex
                self.safely_execute("session_timing_features", "Session timing features")
                self.safely_execute("session_transition_features", "Session Transition Features")
                self.safely_execute("seasonal_decompose", "Seasonal Decomposition")
                self.safely_execute("stl_decomposition", "STL Decomposition")
                self.safely_execute("mstl_decomposition", "MSTL Decomposition")
                self.safely_execute("holiday_features", "Holiday Features")
                logging.debug("Finished calculating Time-based Features")
                
                # Market Regime Features
                self.safely_execute("volatility_regimes", "Volatility Regimes")
                self.safely_execute("trend_strength_indicators", "Trend Strength Indicators")
                self.safely_execute("market_state_indicators", "Market State Indicators")
                
                # Feature Transformations
                self.safely_execute("log_returns", "Log Returns")
                self.safely_execute("fast_fourier_transforms", "Fast Fourier Transforms")
            
            try:
                n_features, feature_list = self._generated_features()
                logging.info(f"Generated {n_features} features.")
                logging.debug(f"Generated features are: {feature_list}")
            except Exception as e:
                logging.error(f"Error calculating feature statistics: {str(e)}")
                
        except Exception as e:
            logging.error(f"Unexpected error during feature generation: {str(e)}")
            
        finally:
            # Always return the DataFrame, even if some features failed to generate
            return self.df
    
    def safely_execute(self, method_name, feature_name):
        """Safely execute and time a feature generation method with proper error handling."""
        try:
            start = time.time()
            method = getattr(self, method_name)
            method()
            end = time.time()
            logging.debug(f"Feature {feature_name} took {end - start} seconds to calculate.")
            return True
        except Exception as e:
            logging.error(f"Error generating {feature_name}: {str(e)}")
            return False
    
    def _generated_features(self, basic_cols = ['date', 'open', 'high', 'low', 'close', 'volume']):
        """
        Function for counting generated features, assuming that's whats in the dataset.
        
        Parameters:
            basic_cols: list of the columns that are not features, but basic columns. Default is date, OHLCV
            
        Returns:
            n_features: Number of features, excluding basic_cols
            features_list: List of all features.
        """
        columns = self.df.columns
        
        features_list = [col for col in columns if col not in basic_cols]
        
        return len(features_list), features_list
    
    def _add_features_efficiently(self, new_features_dict, drop_columns=None):
        """
        Helper method to add multiple features efficiently without DataFrame fragmentation.
        
        Parameters:
        -----------
        new_features_dict : dict
            Dictionary where keys are column names and values are Series/arrays with the feature data
        drop_columns : list, optional
            List of column names to drop after adding new features (for intermediate calculations)
        """
        if not new_features_dict:
            return
            
        # Create DataFrame from new features
        new_features_df = pd.DataFrame(new_features_dict, index=self.df.index)
        
        # Concatenate with original DataFrame
        self.df = pd.concat([self.df, new_features_df], axis=1)
        
        # Drop intermediate columns if specified
        if drop_columns:
            existing_drop_cols = [col for col in drop_columns if col in self.df.columns]
            if existing_drop_cols:
                self.df = self.df.drop(columns=existing_drop_cols)
    
    # =============================================================================
    # Section: Price Action Features
    # =============================================================================
    
    def moving_averages(self):
        """Generating/calculating both simple and exponential moving averages over different time periods."""
        num_periods = [5, 10, 20, 50, 200]
        
        # Collect all moving averages in a dictionary
        new_features = {}
        
        for period in num_periods:
            new_features[f"sma_{period}"] = self.df["close"].rolling(window=period).mean()
            new_features[f"ema_{period}"] = self.df["close"].ewm(span=period, adjust=False, min_periods=period).mean()
        
        # Add all features at once
        self._add_features_efficiently(new_features)
        
    def rate_of_change(self):
        """Price momentum: Rate of change (ROC) over various lookback periods."""
        lookback_periods = [1, 5, 10, 20, 60]
        
        new_features = {}
        
        for period in lookback_periods:
            # Standard ROC formula: [(Close_t / Close_t-n) - 1] * 100
            new_features[f"roc_{period}"] = (self.df["close"] / self.df["close"].shift(period) - 1) * 100
        
        self._add_features_efficiently(new_features)

    
    def average_true_range(self, period_atr:int = 14):
        """Average of true ranges over specified period. ATR measures volatility, taking into account any gaps in the price movement"""
        
        # Calculate all components without adding to DataFrame
        high_low = self.df["high"] - self.df["low"]
        high_prev_close = abs(self.df["high"] - self.df["close"].shift(1))
        low_prev_close = abs(self.df["low"] - self.df["close"].shift(1))
        
        # Calculate true range using vectorized max
        true_range_df = pd.DataFrame({
            'high_low': high_low,
            'high_prev_close': high_prev_close,
            'low_prev_close': low_prev_close
        })
        true_range = true_range_df.max(axis=1)
        
        # Calculate ATR
        atr = true_range.ewm(span=period_atr, adjust=False).mean()
        
        # Add only the final ATR feature
        self._add_features_efficiently({"atr": atr})

    def bollinger_bands(self, period_bb:int = 20):
        """Statistical chart characterizing the prices and volatility over time. Represent standard deviation channels around price"""
        
        # Calculate all components
        sma_key = f'sma_{period_bb}'
        std = self.df["close"].rolling(window=period_bb).std()
        num_std_dev = 2
        
        # Check if SMA already exists, if not calculate it
        if sma_key not in self.df.columns:
            sma = self.df["close"].rolling(window=period_bb).mean()
        else:
            sma = self.df[sma_key]
        
        # Calculate bands
        upper_band = sma + (num_std_dev * std)
        lower_band = sma - (num_std_dev * std)
        bb_width = (upper_band - lower_band) / sma
        
        # Prepare new features
        new_features = {
            'upper_band': upper_band,
            'lower_band': lower_band,
            'bb_width': bb_width
        }
        
        # Add SMA if it wasn't already present
        if sma_key not in self.df.columns:
            new_features[sma_key] = sma
            
        self._add_features_efficiently(new_features)
        
    def support_resistance(self, lookback=20):
        """Distance from recent high/lows."""
        new_features = {
            'distance_from_high': (self.df['close'] / self.df['high'].rolling(lookback).max() - 1) * 100,
            'distance_from_low': (self.df['close'] / self.df['low'].rolling(lookback).min() - 1) * 100
        }
        
        self._add_features_efficiently(new_features)
        
        # TODO: add "distance from pivot points"
        
    # =============================================================================
    # Section: Volume-Based Features
    # =============================================================================
    def volume_moving_averages(self):
        """Identify unusual volume spikes."""
        num_periods = [7, 15, 30, 60]
        
        new_features = {}
        
        for period in num_periods:
            vma = self.df['volume'].rolling(window=period).mean()
            new_features[f'vma_{period}'] = vma
            new_features[f'relative_volume_{period}'] = self.df['volume'] / vma
            
        self._add_features_efficiently(new_features)
            
    def volume_rate_of_change(self):
        """How rapidly volume is increasing/decreasing."""
        lookback_periods = [1, 5, 10, 20]
        
        new_features = {}
        
        for period in lookback_periods:
            new_features[f"vroc_{period}"] = (self.df["volume"] / self.df["volume"].shift(period) - 1) * 100
        
        self._add_features_efficiently(new_features)
        
    def on_balance_volume(self):
        """Cumulative indicator that relates volume to price changes."""
        # Calculate price direction
        price_direction = np.sign(self.df['close'].diff())
        # Set direction to 0 for unchanged prices
        price_direction[self.df['close'].diff() == 0] = 0
        # Calculate OBV
        volume_direction = self.df['volume'] * price_direction
        obv = volume_direction.cumsum()
        
        self._add_features_efficiently({'obv': obv})
        
    def volume_profile(self, num_bins=10, lookback=20, key_metrics=True):
        """
        Calculate rolling volume profile with adaptive bins and key metrics.

        Volume Profile shows the distribution of trading volume across price levels,
        identifying areas of high interest/liquidity in the market.
        
        Parameters:
        -----------
        num_bins : int
            Number of price bins to divide the range into
        lookback : int
            Number of periods to include in each rolling calculation
        key_metrics : bool
            Whether to calculate key volume profile metrics (POC, value area, etc.)
        """
        # Pre-allocate arrays for optimization
        vol_profiles = np.zeros((len(self.df), num_bins))
        
        # Vectorized calculation of min/max for each window
        rolling_min = self.df['low'].rolling(window=lookback).min()
        rolling_max = self.df['high'].rolling(window=lookback).max()
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        # Initialize new features dictionary
        new_features = {}
        
        # Add typical_price temporarily for calculations
        temp_df = self.df.copy()
        temp_df['typical_price'] = typical_price
        
        # Initialize arrays for key metrics
        vol_poc = np.full(len(self.df), np.nan)
        vol_va_high = np.full(len(self.df), np.nan)
        vol_va_low = np.full(len(self.df), np.nan)
        vol_va_width = np.full(len(self.df), np.nan)
        
        # Process each window
        for i in range(lookback, len(self.df)):
            window = temp_df.iloc[i-lookback:i]
            price_min = rolling_min.iloc[i]
            price_max = rolling_max.iloc[i]
            price_range = price_max - price_min
            
            if price_range > 0:
                bin_size = price_range / num_bins
                
                # Calculate volume distribution
                for j, row in window.iterrows():
                    bin_idx = min(num_bins - 1, int((row['typical_price'] - price_min) / bin_size))
                    time_factor = 1 + 0.1 * (window.index.get_loc(j) / lookback)
                    vol_profiles[i, bin_idx] += row['volume'] * time_factor
                
                # Normalize
                total_vol = np.sum(vol_profiles[i])
                if total_vol > 0:
                    vol_profiles[i] = vol_profiles[i] / total_vol * 100
                
                # Add volume profile bins to features
                for bin_num in range(num_bins):
                    if f'vol_bin_{bin_num}' not in new_features:
                        new_features[f'vol_bin_{bin_num}'] = np.full(len(self.df), np.nan)
                    new_features[f'vol_bin_{bin_num}'][i] = vol_profiles[i, bin_num]
                
                # Calculate key metrics if requested
                if key_metrics:
                    poc_bin = np.argmax(vol_profiles[i])
                    vol_poc[i] = price_min + (poc_bin + 0.5) * bin_size
                    
                    # Value Area calculation
                    value_area_threshold = 0.7
                    sorted_bins = np.argsort(vol_profiles[i])[::-1]
                    cum_vol = 0
                    value_area_bins = []
                    
                    for bin_idx in sorted_bins:
                        value_area_bins.append(bin_idx)
                        cum_vol += vol_profiles[i, bin_idx] / 100
                        if cum_vol >= value_area_threshold:
                            break
                    
                    vol_va_high[i] = price_min + (max(value_area_bins) + 1) * bin_size
                    vol_va_low[i] = price_min + min(value_area_bins) * bin_size
                    vol_va_width[i] = (vol_va_high[i] - vol_va_low[i]) / price_range
        
        # Convert arrays to Series and add to features
        if key_metrics:
            new_features['vol_poc'] = pd.Series(vol_poc, index=self.df.index)
            new_features['vol_va_high'] = pd.Series(vol_va_high, index=self.df.index)
            new_features['vol_va_low'] = pd.Series(vol_va_low, index=self.df.index)
            new_features['vol_va_width'] = pd.Series(vol_va_width, index=self.df.index)
            
            # Calculate derivative features
            close_to_poc_pct = (self.df['close'] - new_features['vol_poc']) / new_features['vol_poc'] * 100
            in_value_area = ((self.df['close'] >= new_features['vol_va_low']) & 
                            (self.df['close'] <= new_features['vol_va_high'])).astype(int)
            
            # Volume concentration (entropy-based)
            entropy = np.zeros(len(self.df))
            for i in range(lookback, len(self.df)):
                profile = vol_profiles[i] / 100
                profile = profile[profile > 0]
                if len(profile) > 0:
                    entropy[i] = -np.sum(profile * np.log(profile))
            vol_concentration = 1 - entropy / np.log(num_bins)
            
            new_features['close_to_poc_pct'] = close_to_poc_pct
            new_features['in_value_area'] = in_value_area
            new_features['vol_concentration'] = pd.Series(vol_concentration, index=self.df.index)
        
        # Convert any remaining numpy arrays to Series
        for key, value in new_features.items():
            if isinstance(value, np.ndarray):
                new_features[key] = pd.Series(value, index=self.df.index)
        
        self._add_features_efficiently(new_features)
    
    # FX doesn't have traditional volume, so you might use tick volume or spread-based proxies
    def fx_activity_profile(self, lookback=20):
        """Create activity profile using price movement and spread changes"""
        # Use high-low range as activity proxy
        self.df['activity'] = (self.df['high'] - self.df['low']) / self.df['close']
    
    def vwap(self, period=1):
        """Calculate daily Volume Weighted Average Price."""
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        vwap = (self.df['volume'] * typical_price).rolling(period).sum() / self.df['volume'].rolling(period).sum()
        
        self._add_features_efficiently({'vwap': vwap})
        
    def money_flow_index(self, period=14):
        """Calculate Money Flow Index - a volume-weighted RSI."""
        # Calculate all components without intermediate DataFrame assignments
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        money_flow = typical_price * self.df['volume']
        
        # Positive and negative money flow
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        
        # Money flow sums
        positive_flow_sum = pd.Series(positive_flow, index=self.df.index).rolling(window=period).sum()
        negative_flow_sum = pd.Series(negative_flow, index=self.df.index).rolling(window=period).sum()
        
        # Money flow ratio and index
        money_ratio = np.where(negative_flow_sum != 0, positive_flow_sum / negative_flow_sum, 0)
        mfi = 100 - (100 / (1 + money_ratio))
        
        self._add_features_efficiently({'mfi': mfi})
    
    # =============================================================================
    # Section: Technical Indicators
    # =============================================================================
    def relative_strength_index(self, window=14):
        """Calculate RSI using Wilder's smoothing method."""
        delta = self.df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # First values
        avg_gain = gain[:window].mean()
        avg_loss = loss[:window].mean()
        
        # Initialize lists with first values
        avg_gains = [avg_gain]
        avg_losses = [avg_loss]
        
        # Apply Wilder's smoothing
        for i in range(window, len(gain)):
            avg_gain = ((avg_gains[-1] * (window - 1)) + gain.iloc[i]) / window
            avg_loss = ((avg_losses[-1] * (window - 1)) + loss.iloc[i]) / window
            avg_gains.append(avg_gain)
            avg_losses.append(avg_loss)
        
        # Convert to Series with proper index
        avg_gain_series = pd.Series(avg_gains, index=self.df.index[window-1:])
        avg_loss_series = pd.Series(avg_losses, index=self.df.index[window-1:])
        
        # Calculate RS and RSI
        rs = avg_gain_series / avg_loss_series
        rsi = 100 - (100 / (1 + rs))
        
        # Add to dataframe
        rsi_full = pd.Series(index=self.df.index, dtype=float)
        rsi_full.loc[rsi.index] = rsi
        
        self._add_features_efficiently({f'rsi_{window}': rsi_full})
    
    def moving_average_convergence_divergence(self):
        """Trend-following momentum indicator."""
        # Calculate EMAs
        ema_12 = self.df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = self.df['close'].ewm(span=26, adjust=False).mean()

        # Calculate MACD components
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - macd_signal
        
        new_features = {
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }
        
        self._add_features_efficiently(new_features)
    
    def average_directional_index(self, lookback=14):
        """Measures trend strength."""
        # Calculate +DM and -DM
        high_diff = self.df["high"].diff()
        low_diff = self.df["low"].diff()
        
        # True +DM and -DM conditions
        up_move = high_diff
        down_move = -low_diff
        
        # +DM is up_move if up_move > down_move and up_move > 0, otherwise 0
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        # -DM is down_move if down_move > up_move and down_move > 0, otherwise 0
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate True Range
        tr1 = self.df["high"] - self.df["low"]
        tr2 = abs(self.df["high"] - self.df["close"].shift(1))
        tr3 = abs(self.df["low"] - self.df["close"].shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing
        smoothed_tr = tr.rolling(window=lookback).sum()
        smoothed_plus_dm = pd.Series(plus_dm).rolling(window=lookback).sum()
        smoothed_minus_dm = pd.Series(minus_dm).rolling(window=lookback).sum()
        
        # For subsequent calculations after the initial lookback period
        for i in range(lookback, len(tr)):
            smoothed_tr.iloc[i] = smoothed_tr.iloc[i-1] - (smoothed_tr.iloc[i-1]/lookback) + tr.iloc[i]
            smoothed_plus_dm.iloc[i] = smoothed_plus_dm.iloc[i-1] - (smoothed_plus_dm.iloc[i-1]/lookback) + plus_dm[i]
            smoothed_minus_dm.iloc[i] = smoothed_minus_dm.iloc[i-1] - (smoothed_minus_dm.iloc[i-1]/lookback) + minus_dm[i]
        
        # Calculate +DI and -DI
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        # Calculate ADX - smoothed DX
        adx = dx.rolling(window=lookback).mean()
        
        # Add all features at once
        new_features = {
            "+di": plus_di,
            "-di": minus_di,
            "adx": adx
        }
        
        self._add_features_efficiently(new_features)
        
    def stochastic_oscillator(self, k_period=14, d_period=3, slowing=3):
        """Calculate Stochastic Oscillator."""
        # Find the lowest low and highest high for the k_period
        low_min = self.df['low'].rolling(window=k_period).min()
        high_max = self.df['high'].rolling(window=k_period).max()
        
        # Calculate %K (Fast Stochastic)
        k_fast = 100 * (self.df['close'] - low_min) / (high_max - low_min)
        
        # Handle division by zero
        k_fast = k_fast.replace([np.inf, -np.inf], np.nan).fillna(50)
        
        # Apply slowing period to fast %K 
        k = k_fast.rolling(window=slowing).mean() if slowing > 1 else k_fast
        
        # Calculate %D (signal line) - simple moving average of %K
        d = k.rolling(window=d_period).mean()
        
        new_features = {
            'stoch_k': k,
            'stoch_d': d
        }
        
        self._add_features_efficiently(new_features)
        
    def commodity_channel_index(self, period=20):
        """Calculate Commodity Channel Index."""
        # Typical Price
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        # Moving Average of Typical Price
        ma_tp = tp.rolling(window=period).mean()
        # Mean Absolute Deviation (MAD)
        mad = abs(tp - ma_tp).rolling(window=period).mean()
        # CCI
        cci = (tp - ma_tp) / (0.015 * mad)
        
        self._add_features_efficiently({'cci': cci})
    
    def ichimoku_cloud(self, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
        """Calculate Ichimoku Cloud components."""
        # Calculate all components
        tenkan_sen = (self.df['high'].rolling(window=tenkan_period).max() + 
                    self.df['low'].rolling(window=tenkan_period).min()) / 2
        
        kijun_sen = (self.df['high'].rolling(window=kijun_period).max() + 
                    self.df['low'].rolling(window=kijun_period).min()) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        senkou_span_b = ((self.df['high'].rolling(window=senkou_b_period).max() + 
                        self.df['low'].rolling(window=senkou_b_period).min()) / 2).shift(displacement)
        
        chikou_span = self.df['close'].shift(-displacement)
        
        # Add all features at once
        new_features = {
            'ichimoku_tenkan': tenkan_sen,
            'ichimoku_kijun': kijun_sen,
            'ichimoku_senkou_a': senkou_span_a,
            'ichimoku_senkou_b': senkou_span_b,
            'ichimoku_chikou': chikou_span
        }
        
        self._add_features_efficiently(new_features)
        
    # =============================================================================
    # Section: Time-based Features
    # =============================================================================
    def time_of_day(self):
        """Extract time of day features from datetime index."""
        # Extract hour of day
        hour = self.df.index.hour
        
        all_features, values = data_utils.check_feature_specification(self.selected_features, 'time_of_day')
        
        new_features = {}
        
        # Create cyclical features using sine and cosine transformations
        if all_features or not all_features and 'hour_sin' in values:
            new_features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        if all_features or not all_features and 'hour_cos' in values:
            new_features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
        self._add_features_efficiently(new_features)
    
    def time_of_week(self):
        """Extract time of week features from datetime index."""
        day_of_week = self.df.index.dayofweek
        hour_of_day = self.df.index.hour
        hour_of_week = day_of_week * 24 + hour_of_day
        
        new_features = {
            'hour_of_week_sin': np.sin(2 * np.pi * hour_of_week / 168),
            'hour_of_week_cos': np.cos(2 * np.pi * hour_of_week / 168),
            'is_weekend': (day_of_week >= 5).astype(int)
        }
        
        self._add_features_efficiently(new_features)
    
    def time_of_month(self):
        """Extract time of month features from datetime index."""
        day = self.df.index.day
        days_in_month = pd.Series(self.df.index).dt.days_in_month.values
        normalized_day = day / days_in_month
        month = self.df.index.month
        
        # Create month period categories
        month_period = pd.cut(
            normalized_day, 
            bins=[0, 0.33, 0.66, 1],
            labels=['start_of_month', 'mid_month', 'end_of_month']
        )
        
        new_features = {
            'month_progress_sin': np.sin(2 * np.pi * normalized_day),
            'month_progress_cos': np.cos(2 * np.pi * normalized_day),
            'month_period': month_period,
            'month_of_year_sin': np.sin(2 * np.pi * month / 12),
            'month_of_year_cos': np.cos(2 * np.pi * month / 12)
        }
        
        self._add_features_efficiently(new_features)
    
    def day_of_week(self):
        """Extract day of week features from datetime index."""
        day_of_week = self.df.index.dayofweek  # Monday=0, Sunday=6
        
        new_features = {
            # Standard cyclical features
            'day_of_week_sin': np.sin(2 * np.pi * day_of_week / 7),
            'day_of_week_cos': np.cos(2 * np.pi * day_of_week / 7),
            
            # FX-specific day flags
            'is_monday': (day_of_week == 0).astype(int),
            'is_friday': (day_of_week == 4).astype(int),
            'is_midweek': (day_of_week.isin([1, 2, 3])).astype(int),
            
            # Weekend proximity
            'friday_afternoon': ((day_of_week == 4) & (self.df.index.hour >= 15)).astype(int),
            'sunday_evening': ((day_of_week == 6) & (self.df.index.hour >= 20)).astype(int)
        }
        
        self._add_features_efficiently(new_features)
    
    def fx_trading_sessions(self):
        """Create proper FX trading session features with overlaps."""
        hour = self.df.index.hour
        
        # Calculate all session indicators
        asian_session = ((hour >= 22) | (hour < 8)).astype(int)
        london_session = ((hour >= 7) & (hour < 16)).astype(int)
        ny_session = ((hour >= 12) & (hour < 21)).astype(int)
        
        # Session overlaps
        asian_london_overlap = ((hour >= 7) & (hour < 8)).astype(int)
        london_ny_overlap = ((hour >= 12) & (hour < 16)).astype(int)
        
        # Primary session
        conditions = [
            (hour >= 12) & (hour < 16),
            (hour >= 7) & (hour < 12),
            (hour >= 16) & (hour < 21),
            ((hour >= 22) | (hour < 7)),
            True
        ]
        choices = ['london_ny_overlap', 'london_primary', 'ny_primary', 'asian_primary', 'off_hours']
        primary_session = np.select(conditions, choices)
        
        # Session intensity mapping
        session_intensity_map = {
            'london_ny_overlap': 1.0,
            'london_primary': 0.8,
            'ny_primary': 0.7,
            'asian_primary': 0.6,
            'off_hours': 0.2
        }
        session_intensity = pd.Series(primary_session).map(session_intensity_map).values
        
        new_features = {
            'asian_session': asian_session,
            'london_session': london_session,
            'ny_session': ny_session,
            'asian_london_overlap': asian_london_overlap,
            'london_ny_overlap': london_ny_overlap,
            'primary_session': primary_session,
            'session_intensity': session_intensity
        }
        
        self._add_features_efficiently(new_features)
    
    def session_timing_features(self):
        """Add granular timing features within sessions"""
        hour = self.df.index.hour
        minute = self.df.index.minute
        
        # Minutes since session start/end
        def minutes_since_session_start(row):
            hour, minute = row.name.hour, row.name.minute
            current_minutes = hour * 60 + minute
            
            # London session start: 7:00 GMT
            if hour >= 7 and hour < 16:
                london_start = 7 * 60  # 7:00 in minutes
                return current_minutes - london_start
            
            # NY session start: 12:00 GMT  
            elif hour >= 12 and hour < 21:
                ny_start = 12 * 60  # 12:00 in minutes
                return current_minutes - ny_start
                
            # Asian session (wraps around midnight)
            elif hour >= 22 or hour < 8:
                if hour >= 22:
                    asian_start = 22 * 60  # 22:00 in minutes
                    return current_minutes - asian_start
                else:
                    # After midnight
                    asian_start = 22 * 60  # Previous day 22:00
                    return (current_minutes + 24*60) - asian_start
            
            return 0
        
        self.df['minutes_since_session_start'] = self.df.apply(minutes_since_session_start, axis=1)
        
        # Session progress (0-1, how far through the session)
        session_lengths = {
            'london_primary': 9 * 60,      # 9 hours
            'ny_primary': 9 * 60,          # 9 hours  
            'asian_primary': 10 * 60,      # 10 hours
            'london_ny_overlap': 4 * 60,   # 4 hours
            'off_hours': 1 * 60            # 1 hour (arbitrary)
        }
        
        self.df['session_progress'] = self.df.apply(
            lambda row: min(1.0, row['minutes_since_session_start'] / 
                          session_lengths.get(row['primary_session'], 60)), axis=1
        )
        
        # Cyclical session progress
        progress = self.df['session_progress']
        self.df['session_progress_sin'] = np.sin(2 * np.pi * progress)
        self.df['session_progress_cos'] = np.cos(2 * np.pi * progress)
    
    def session_transition_features(self):
        """Features for session handoffs and transitions"""
        hour = self.df.index.hour
        minute = self.df.index.minute
        
        # Minutes until next major session
        def minutes_to_next_session(row):
            hour, minute = row.name.hour, row.name.minute
            current_minutes = hour * 60 + minute
            
            # Next session start times (in minutes from midnight)
            session_starts = [7*60, 12*60, 16*60, 21*60, 22*60, (7+24)*60]  # Include next day London
            
            for start_time in session_starts:
                if start_time > current_minutes:
                    return start_time - current_minutes
            
            # If past all sessions today, next is tomorrow's London
            return (7*60 + 24*60) - current_minutes
        
        self.df['minutes_to_next_session'] = self.df.apply(minutes_to_next_session, axis=1)
        
        # Session transition flags (30 minutes before/after session starts)
        transition_window = 30  # minutes
        self.df['session_transition'] = (
            (self.df['minutes_to_next_session'] <= transition_window) |
            (self.df['minutes_since_session_start'] <= transition_window)
        ).astype(int)
        
        # Specific transition periods
        self.df['pre_london'] = ((hour == 6) & (minute >= 30) | (hour == 7) & (minute < 30)).astype(int)
        self.df['pre_ny'] = ((hour == 11) & (minute >= 30) | (hour == 12) & (minute < 30)).astype(int)
        self.df['london_close'] = ((hour == 15) & (minute >= 30) | (hour == 16) & (minute < 30)).astype(int)
    
    def seasonal_decompose(self, period=160):
        """
        Perform seasonal decomposition on the closing price.
        Adds trend, seasonal, and residual components to the dataframe.
        """
        try:
            # We need to make sure there are no NaN values
            close_series = self.df['close'].dropna()
            
            # Skip if we don't have enough data
            if len(close_series) < 2 * period:
                logging.warning(f"Not enough data for seasonal decomposition (need {2*period}, have {len(close_series)})")
                return
            
            # Perform decomposition
            result = seasonal_decompose(
                close_series, 
                model='additive', 
                period=period, 
                extrapolate_trend='freq'
            )
            
            # Add components to the dataframe
            self.df['seasonal_trend'] = result.trend
            self.df['seasonal_seasonal'] = result.seasonal
            self.df['seasonal_residual'] = result.resid
            
            # Add a normalized seasonal component (percentage of price)
            self.df['seasonal_norm'] = self.df['seasonal_seasonal'] / self.df['close']
            
        except Exception as e:
            logging.warning(f"Error calculating seasonal decompose: {e}")
            
    def stl_decomposition(self, period=160):
        """
        Perform STL decomposition on the closing price.
        STL is more robust to outliers than classical decomposition.
        """
        try:
            # We need to make sure there are no NaN values
            close_series = self.df['close'].dropna()
            
            # Skip if we don't have enough data
            if len(close_series) < 2 * period:
                logging.warning(f"Not enough data for STL decomposition (need {2*period}, have {len(close_series)})")
                return
            
            # Perform STL decomposition
            stl = STL(close_series, period=period, seasonal=13)
            result = stl.fit()
            
            # Add components to the dataframe
            self.df['stl_trend'] = result.trend
            self.df['stl_seasonal'] = result.seasonal
            self.df['stl_residual'] = result.resid
            
            # Add seasonal strength metrics
            self.df['stl_seasonal_strength'] = 1 - (result.resid.var() / (result.seasonal + result.resid).var())
            self.df['stl_trend_strength'] = 1 - (result.resid.var() / (result.trend + result.resid).var())
            
        except Exception as e:
            logging.warning(f"Error calculating STL decomposition: {e}")
            
    def mstl_decomposition(self, periods=[24, 160]):
        """
        Perform MSTL decomposition for multiple seasonality components.
        Good for data with intraday and multi-day patterns.
        """
        try:
            # We need to make sure there are no NaN values
            close_series = self.df['close'].dropna()
            
            # Skip if we don't have enough data
            if len(close_series) < 2 * max(periods):
                logging.warning(f"Not enough data for MSTL decomposition (need {2*max(periods)}, have {len(close_series)})")
                return
            
            # Perform MSTL decomposition
            mstl = MSTL(close_series, periods=periods)
            result = mstl.fit()
            
            # Add components to the dataframe
            self.df['mstl_trend'] = result.trend
            
            # Add each seasonal component
            for i, period in enumerate(periods):
                self.df[f'mstl_seasonal_{period}'] = result.seasonal_[i]
            
            self.df['mstl_residual'] = result.resid
            
        except Exception as e:
            logging.warning(f"Error calculating MSTL decomposition: {e}")
            
    def holiday_features(self):
        """
        Add features for market holidays and special events.
        Requires holidays package: pip install holidays
        """
        try:
            import holidays
            
            # Get date from the index
            dates = pd.Series(self.df.index.date)
            
            # Create a holidays calendar (adjust for your target market)
            us_holidays = holidays.US(years=range(dates.dt.year.min(), dates.dt.year.max()+1))
            
            # Flag for holidays
            self.df['is_holiday'] = dates.isin(us_holidays).astype(int)
            
            # Distance to next holiday (market behavior often changes approaching holidays)
            next_holiday = np.zeros(len(self.df))
            for i, date in enumerate(dates):
                # Find next holiday
                for days_ahead in range(1, 30):
                    future_date = date + pd.Timedelta(days=days_ahead)
                    if future_date in us_holidays:
                        next_holiday[i] = days_ahead
                        break
            self.df['days_to_next_holiday'] = next_holiday
            
        except ImportError:
            logging.warning("holidays package not installed. Run 'pip install holidays'")
        except Exception as e:
            logging.warning(f"Error calculating holiday features: {e}")
    
    # =============================================================================
    # Section: Market Regime Features
    # =============================================================================
    def volatility_regimes(self, lookback_short=20, lookback_long=100, n_regimes=3):
        """
        Identify volatility regimes (high, medium, low) based on historical volatility.
        
        Parameters:
        -----------
        lookback_short : int
            Window for short-term volatility calculation
        lookback_long : int
            Window for longer-term volatility context
        n_regimes : int
            Number of volatility regimes to identify (typically 2 or 3)
        """
        # Calculate rolling volatility (standard deviation of returns)
        returns = self.df['close'].pct_change()
        
        # Create all new columns in a dictionary first
        new_columns = {}
        
        # Calculate volatility components
        volatility_short = returns.rolling(window=lookback_short).std() * np.sqrt(252)  # Annualized
        volatility_long = returns.rolling(window=lookback_long).std() * np.sqrt(252)
        volatility_long = volatility_long.replace(0, np.nan)
        
        # Add epsilon to avoid division by extremely small numbers
        epsilon = 1e-8
        volatility_ratio = volatility_short / (volatility_long + epsilon)
        
        # Store in dictionary
        new_columns['returns'] = returns
        new_columns['volatility_short'] = volatility_short
        new_columns['volatility_long'] = volatility_long
        new_columns['volatility_ratio'] = volatility_ratio
        
        # Identify volatility regimes using quantiles
        if n_regimes == 2:
            # Simple binary regime (high/low)
            volatility_regime = np.where(
                volatility_short > volatility_short.rolling(window=lookback_long).quantile(0.5),
                'high', 'low'
            )
        else:
            # Multi-regime approach (e.g., low/medium/high)
            # Calculate historical percentiles for current volatility
            vol_percentile = volatility_short.rolling(window=lookback_long).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
            
            # Assign regimes based on percentiles
            conditions = [
                (vol_percentile < 0.33),
                (vol_percentile >= 0.33) & (vol_percentile < 0.66),
                (vol_percentile >= 0.66)
            ]
            choices = ['low', 'medium', 'high']
            volatility_regime = np.select(conditions, choices, default='medium')
        
        new_columns['volatility_regime'] = volatility_regime
        
        # One-hot encode the regime
        regime_dummies = pd.get_dummies(pd.Series(volatility_regime, index=self.df.index), prefix='vol_regime')
        
        # Add continuous volatility Z-score
        vol_mean = volatility_short.rolling(window=lookback_long).mean()
        vol_std = volatility_short.rolling(window=lookback_long).std()
        new_columns['volatility_zscore'] = (volatility_short - vol_mean) / vol_std
        
        # Volatility acceleration/deceleration
        new_columns['volatility_change'] = np.log(volatility_short + epsilon).diff(5)
        
        # Create DataFrame from new columns and concatenate all at once
        new_df = pd.DataFrame(new_columns, index=self.df.index)
        new_df = pd.concat([new_df, regime_dummies], axis=1)
        
        # Concatenate with original DataFrame
        self.df = pd.concat([self.df, new_df], axis=1)

    def trend_strength_indicators(self, lookback=20, adx_threshold=25):
        """Create indicators to identify trending vs. ranging markets."""
        new_features = {}
        
        # 1. ADX-based trend regime (if ADX exists)
        if 'adx' in self.df.columns:
            adx_trend_regime = np.where(self.df['adx'] > adx_threshold, 'trending', 'ranging')
            new_features['adx_trend_regime'] = adx_trend_regime
            
            # One-hot encode
            regime_dummies = pd.get_dummies(pd.Series(adx_trend_regime, index=self.df.index), prefix='adx_regime')
            for col in regime_dummies.columns:
                new_features[col] = regime_dummies[col]
        
        # 2. Price vs Moving Average indicators
        for period in [20, 50, 200]:
            sma_col = f'sma_{period}'
            if sma_col not in self.df.columns:
                new_features[sma_col] = self.df['close'].rolling(window=period).mean()
                sma_values = new_features[sma_col]
            else:
                sma_values = self.df[sma_col]
            
            # Distance from SMA and slope
            new_features[f'dist_from_sma_{period}'] = (self.df['close'] / sma_values - 1) * 100
            new_features[f'sma_{period}_slope'] = sma_values.pct_change(5, fill_method=None) * 100
        
        # 3. Moving average crossovers
        if 'sma_50' in self.df.columns and 'sma_200' in self.df.columns:
            new_features['ma_cross_50_200'] = np.where(self.df['sma_50'] > self.df['sma_200'], 1, -1)
        elif 'sma_50' in new_features and 'sma_200' in new_features:
            new_features['ma_cross_50_200'] = np.where(new_features['sma_50'] > new_features['sma_200'], 1, -1)
        
        # 4. Linear regression indicators
        regression_slope = np.full(len(self.df), np.nan)
        r_squared = np.full(len(self.df), np.nan)
        
        for i in range(lookback, len(self.df)):
            y = self.df['close'].iloc[i-lookback:i].values
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
            regression_slope[i] = slope
            r_squared[i] = r_value**2
        
        new_features['regression_slope'] = pd.Series(regression_slope, index=self.df.index)
        new_features['r_squared'] = pd.Series(r_squared, index=self.df.index)
        new_features['norm_regression_slope'] = new_features['regression_slope'] / self.df['close']
        
        # 5. Combined trend strength indicator
        components = []
        
        if 'adx' in self.df.columns:
            components.append(self.df['adx'] / 100)
        
        components.append(new_features['r_squared'])
        
        # MA alignment component
        ma_alignment = (
            np.sign(new_features['dist_from_sma_20']) * 
            np.sign(new_features['dist_from_sma_50']) * 
            np.sign(new_features['dist_from_sma_200'])
        )
        ma_aligned = (abs(ma_alignment) == 1).astype(float)
        components.append(ma_aligned)
        
        trend_strength = pd.concat(components, axis=1).mean(axis=1) * 100
        new_features['trend_strength'] = trend_strength
        
        # 6. Trend regime classification
        trend_regime = np.where(
            trend_strength > 70, 'strong_trend',
            np.where(trend_strength > 40, 'weak_trend', 'ranging')
        )
        new_features['trend_regime'] = trend_regime
        
        # One-hot encode trend regime
        trend_dummies = pd.get_dummies(pd.Series(trend_regime, index=self.df.index), prefix='trend_regime')
        for col in trend_dummies.columns:
            new_features[col] = trend_dummies[col]
        
        self._add_features_efficiently(new_features)
    
    def market_state_indicators(self, window=20):
        """
        Generate overall market state indicators that combine volatility and trend regimes.
        Note: This depends on volatility regime existing in the df. TODO: Add better handling for that.
        
        Parameters:
        -----------
        window : int
            Lookback window for calculations
        """
        # Ensure we have both volatility and trend regimes
        if 'volatility_regime' not in self.df.columns or 'trend_regime' not in self.df.columns:
            logging.warning("Volatility or trend regimes not calculated. Run those methods first.")
            return
        
        # Combine volatility and trend regimes into a consolidated market state
        # Create a mapping of volatility regime * trend regime combinations
        vol_mapping = {'low': 0, 'medium': 1, 'high': 2}
        trend_mapping = {'ranging': 0, 'weak_trend': 1, 'strong_trend': 2}
        
        # Convert string regimes to numeric values if necessary
        if self.df['volatility_regime'].dtype == 'object':
            self.df['vol_regime_num'] = self.df['volatility_regime'].map(vol_mapping)
        
        if self.df['trend_regime'].dtype == 'object':
            self.df['trend_regime_num'] = self.df['trend_regime'].map(trend_mapping)
        
        # Create a combined market state (9 possible states)
        # 0: low vol + ranging, 1: low vol + weak trend, ..., 8: high vol + strong trend
        self.df['market_state'] = self.df['vol_regime_num'] * 3 + self.df['trend_regime_num']
        
        # Map numeric states to descriptive labels
        state_mapping = {
            0: 'low_vol_ranging',
            1: 'low_vol_weak_trend',
            2: 'low_vol_strong_trend',
            3: 'medium_vol_ranging',
            4: 'medium_vol_weak_trend',
            5: 'medium_vol_strong_trend',
            6: 'high_vol_ranging',
            7: 'high_vol_weak_trend',
            8: 'high_vol_strong_trend'
        }
        self.df['market_state_desc'] = self.df['market_state'].map(state_mapping)
        
        # One-hot encode for ML models
        state_dummies = pd.get_dummies(self.df['market_state_desc'], prefix='state')
        self.df = pd.concat([self.df, state_dummies], axis=1)
        
        # Calculate state transition probabilities (useful for predicting regime changes)
        # How stable is the current regime? How likely to switch to another?
        self.df['state_duration'] = 1  # Initialize counter
        
        # Calculate how long we've been in the current state
        current_state = None
        duration = 0
        
        for i in range(len(self.df)):
            if self.df['market_state'].iloc[i] != current_state:
                current_state = self.df['market_state'].iloc[i]
                duration = 1
            else:
                duration += 1
            
            self.df.iloc[i, self.df.columns.get_loc('state_duration')] = duration
    
    # =============================================================================
    # Section: Feature Transformations
    # =============================================================================
    def log_returns(self):
        """Logarithmic returns: measure the percentage change in the price of an asset over a specific period."""
        log_return = np.log(self.df['close'] / self.df['close'].shift(1))
        self._add_features_efficiently({'log_return': log_return})

    def fast_fourier_transforms(self, column='close', n_components=5):
        """
        Extract cyclical components using Fast Fourier Transform.
        
        Parameters:
        -----------
        column : str
            The column to analyze for cyclical patterns
        n_components : int
            Number of dominant frequency components to extract
        """
        # Make sure we work with returns to ensure stationarity
        if 'log_return' not in self.df.columns:
            self._log_returns()
        
        # Get returns data without NaNs
        returns = self.df['log_return'].dropna().values
        
        # Apply FFT
        fft_result = fft(returns)
        
        # Get the power spectrum (absolute value squared)
        power = np.abs(fft_result)**2
        
        # Get frequencies
        sample_freq = fftfreq(len(returns))
        
        # Exclude the DC component (0 frequency)
        positive_freq_idx = np.where((sample_freq > 0))[0]
        freqs = sample_freq[positive_freq_idx]
        powers = power[positive_freq_idx]
        
        # Get indices of the top n_components frequencies
        top_indices = np.argsort(powers)[-n_components:]
        
        # Extract dominant frequencies
        dominant_freqs = freqs[top_indices]
        dominant_periods = 1 / dominant_freqs
        
        # Create features for each dominant cycle
        for i, period in enumerate(dominant_periods):
            # Round to nearest integer for cleaner period values
            period_int = int(round(period))
            
            # Only use reasonable periods (not too short, not too long)
            if 2 <= period_int <= len(returns) // 3:
                # Create sine and cosine features to capture the phase of the cycle
                cycle_sin = np.sin(2 * np.pi * np.arange(len(self.df)) / period_int)
                cycle_cos = np.cos(2 * np.pi * np.arange(len(self.df)) / period_int)
                
                # Add to dataframe
                self.df[f'fft_sin_{period_int}'] = cycle_sin
                self.df[f'fft_cos_{period_int}'] = cycle_cos
        
        # Store the dominant periods as a separate feature
        self.df['fft_dominant_periods'] = str(sorted([int(round(p)) for p in dominant_periods]))        