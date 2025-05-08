import logging
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from volprofile import getVPWithOHLC
from statsmodels.tsa.seasonal import seasonal_decompose, STL , MSTL

from utils import data_utils
from data.features.statistics import FeatureStatistics


class FeatureGenerator():
    def __init__(self, dataset):
        """
        Class focusing on generating features with helpful prediction powers,
        that might help a ML model learn and predict better.
        
        Args:
            dataset: The cleaned (and validated) data, containing (index) date/datetime, and the columns OHLCV.
        
        returns:
            dataset with all generated features 
        """
    
        # Load dataset based on format
        self.df, self.original_df = data_utils.check_and_return_df(dataset)
        
    def run(self):
        """Run feature generation on provided dataset."""
        
        # Price Action Features
        self._moving_averages()
        self._rate_of_change()
        self._average_true_range()  # wrong implementation?
        self._bollinger_bands()  # bad implementation?
        self._support_resistance()  # not sure how to do, yet
        logging.debug("Finished calculating Price Action Features")
        # Volume-Based Features
        self._volume_moving_averages()
        self._volume_rate_of_change()
        self._on_balance_volume()
        self._volume_profile()
        self._vwap()
        self._money_flow_index()
        logging.debug("Finished calculating Volume-Based Features")
        # Technical Indicators
        self._relative_strength_index()  # unsure of impmlementation
        self._moving_average_convergence_divergence()
        self._average_directional_index()  # wrong way to calculate and use??
        self._stochastic_oscillator()
        logging.debug("Finished calculating Technical Indicators")
        # Time-based Features
        self._day_of_week()
        self._seasonal_decompose()  # not properly working or handling for different timeframes
        logging.debug("Finished calculating Time-based Features")
        # Feature Transformations
        self._log_returns()  # Is this implemented right?
        self._normalize_features()
        self._fast_fourier_transforms()
        
        feature_stats = FeatureStatistics(self.df)
        n_features, feature_list = feature_stats.generated_features()
        logging.info(f"Generated {n_features} features.")
        logging.debug(f"Generated features are: {feature_list}")
        
        return self.df
    
    # =============================================================================
    # Section: Price Action Features
    # =============================================================================
    
    def _moving_averages(self):
        """Generating/calculating both simple and exponential moving averages over different time periods."""
        num_periods = [5, 10, 20, 50, 200]
        
        for period in num_periods:
            self.df[f"sma_{period}"] = self.df["close"].rolling(window=period).mean()
            self.df[f"ema_{period}"] = self.df["close"].ewm(span=period, adjust=False, min_periods=period).mean()
        
    def _rate_of_change(self):
        """Price momentum: Rate of change (ROC) over various lookback periods."""
        lookback_periods = [1, 5, 10, 20, 60]
        
        for period in lookback_periods:
            # Standard ROC formula: [(Close_t / Close_t-n) - 1] * 100
            self.df[f"roc_{period}"] = (self.df["close"] / self.df["close"].shift(period) - 1) * 100
    
    def _average_true_range(self, period_atr:int = 14):
        """Average of true ranges over specified period. ATR measures volatility, taking into account any gaps in the price movement"""
        self.df["high_low"] = self.df["high"] - self.df["low"]                          # Current High — Current Low
        self.df["high_prev_close"] = abs(self.df["high"] - self.df["close"].shift(1))   # |Current High — Previous Close|
        self.df["low_prev_close"] = abs(self.df["low"] - self.df["close"].shift(1))     # |Current Low — Previous Close|
        self.df["true_range"] = self.df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
        
        self.df["atr"] = self.df["true_range"].ewm(span=period_atr, adjust=False).mean()
        # Dropping used columns
        self.df = self.df.drop(columns=["high_low", "high_prev_close", "low_prev_close", "true_range"], axis=1)

    def _bollinger_bands(self, period_bb:int = 20):
        """statistical chart characterizing the prices and volatility over time."""
        # middle band is sma
        self.df["std"] = self.df["close"].rolling(window=period_bb).std()  # rolling standard deviation
        num_std_dev = 2
        self.df['upper_band'] = self.df[f'sma_{period_bb}'] + (num_std_dev * self.df['std'])
        self.df['lower_band'] = self.df[f'sma_{period_bb}'] - (num_std_dev * self.df['std'])
        # Add Bollinger Band Width
        self.df['bb_width'] = (self.df['upper_band'] - self.df['lower_band']) / self.df[f'sma_{period_bb}']
        
    def _support_resistance(self, lookback=20):
        """Distance from recent high/lows."""
        # Distance from recent high
        self.df['distance_from_high'] = (self.df['close'] / self.df['high'].rolling(lookback).max() - 1) * 100
        
        # Distance from recent low
        self.df['distance_from_low'] = (self.df['close'] / self.df['low'].rolling(lookback).min() - 1) * 100
        
        # TODO: add "distance from pivot points"
        
    # =============================================================================
    # Section: Volume-Based Features
    # =============================================================================
    def _volume_moving_averages(self):
        """Identify unusual volume spikes."""
        num_periods = [7, 15, 30, 60]
        
        for period in num_periods:
            self.df[f'vma_{period}'] = self.df['volume'].rolling(window=period).mean()
            self.df[f'relative_volume_{period}'] = self.df['volume'] / self.df[f'vma_{period}']
            
    def _volume_rate_of_change(self):
        """How rapidly volume is increasing/decreasing."""
        lookback_periods = [1, 5, 10, 20]
        
        for period in lookback_periods:
            self.df[f"vroc_{period}"] = (self.df["volume"] / self.df["volume"].shift(period) - 1) * 100
        
    def _on_balance_volume(self):
        """Cumulative indicator that relates volume to price changes."""
        # Calculate price direction
        price_direction = np.sign(self.df['close'].diff())
        # Set direction to 0 for unchanged prices
        price_direction[self.df['close'].diff() == 0] = 0
        # Calculate OBV
        volume_direction = self.df['volume'] * price_direction
        self.df['obv'] = volume_direction.cumsum()
        
    def _volume_profile(self, num_bins=10, lookback=20):
        """Calculate rolling volume profile with fixed number of bins."""
        for i in range(lookback, len(self.df)):
            window = self.df.iloc[i-lookback:i]
            price_min = window['low'].min()
            price_max = window['high'].max()
            price_range = price_max - price_min
            
            if price_range > 0:  # Prevent division by zero
                bin_size = price_range / num_bins
                
                for bin_num in range(num_bins):
                    bin_low = price_min + bin_num * bin_size
                    bin_high = bin_low + bin_size
                    
                    # Sum volume where price in this bin (approximation)
                    bin_volume = window.loc[
                        ((window['low'] >= bin_low) & (window['low'] < bin_high)) |
                        ((window['high'] >= bin_low) & (window['high'] < bin_high)) |
                        ((window['low'] <= bin_low) & (window['high'] >= bin_high))
                    ]['volume'].sum()
                    
                    self.df.loc[self.df.index[i], f'vol_bin_{bin_num}'] = bin_volume
    
    def _vwap(self, period=1):
        """Calculate daily Volume Weighted Average Price."""
        # Assuming you have date info in index or as a column
        self.df['vwap'] = (self.df['volume'] * (self.df['high'] + self.df['low'] + self.df['close']) / 3).rolling(period).sum() / self.df['volume'].rolling(period).sum()
        
    def _money_flow_index(self, period=14):
        """Calculate Money Flow Index - a volume-weighted RSI."""
        # Typical price
        self.df['typical_price'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        # Money flow
        self.df['money_flow'] = self.df['typical_price'] * self.df['volume']
        
        # Positive and negative money flow
        self.df['positive_flow'] = np.where(self.df['typical_price'] > self.df['typical_price'].shift(1), 
                                            self.df['money_flow'], 0)
        self.df['negative_flow'] = np.where(self.df['typical_price'] < self.df['typical_price'].shift(1), 
                                            self.df['money_flow'], 0)
        
        # Money flow ratio and index
        self.df['positive_flow_sum'] = self.df['positive_flow'].rolling(window=period).sum()
        self.df['negative_flow_sum'] = self.df['negative_flow'].rolling(window=period).sum()
        
        # Avoid division by zero
        self.df['money_ratio'] = np.where(self.df['negative_flow_sum'] != 0,
                                        self.df['positive_flow_sum'] / self.df['negative_flow_sum'], 0)
        
        self.df['mfi'] = 100 - (100 / (1 + self.df['money_ratio']))
        
        # Drop intermediate columns
        self.df = self.df.drop(['typical_price', 'money_flow', 'positive_flow', 
                            'negative_flow', 'positive_flow_sum', 'negative_flow_sum', 
                            'money_ratio'], axis=1)
    
    # =============================================================================
    # Section: Technical Indicators
    # =============================================================================
    def _relative_strength_index(self, window:int = 14):
        """Measures momentum and overbought/oversold conditions."""
        
        # Calculate price changes
        delta = self.df["close"].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and average loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Add RSI to dataframe
        self.df[f'rsi_{window}'] = rsi
    
    def _moving_average_convergence_divergence(self):
        """Trend-following momentum indicator."""
        self.df['ema_12'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['ema_26'] = self.df['close'].ewm(span=26, adjust=False).mean()

        # Calculate MACD (the difference between 12-period EMA and 26-period EMA)
        self.df['macd'] = self.df['ema_12'] - self.df['ema_26']
        self.df = self.df.drop(["ema_12", "ema_26"], axis=1)  # Don't need that many ema's
        # Calculate the 9-period EMA of MACD (Signal Line)
        #self.df['signal_line'] = self.df['macd'].ewm(span=9, adjust=False).mean()  # not needed?
    
    def _average_directional_index(self, lookback:int = 14):
        """Measures trend strength."""
        plus_dm = self.df["high"].diff()
        minus_dm = self.df["low"].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(self.df["high"] - self.df["low"])
        tr2 = pd.DataFrame(abs(self.df["high"] - self.df["close"].shift(1)))
        tr3 = pd.DataFrame(abs(self.df["low"] - self.df["close"].shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.rolling(lookback).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
        self.df["adx_smooth"] = adx.ewm(alpha = 1/lookback).mean()
        
    def _stochastic_oscillator(self, period=14):
        # Create a copy of existing DataFrame to preserve original index
        # Create new columns for storing values without modifying the index
        self.df['best_low'] = float('nan')
        self.df['best_high'] = float('nan')
        self.df['fast_k'] = float('nan')
        
        # Calculate %K values only where we have enough historical data
        for i in range(period-1, len(self.df)):
            # Get the window of data for the lookback period
            window = self.df.iloc[i-period+1:i+1]
            
            # Find lowest low and highest high in the window
            low = window['close'].min()
            high = window['close'].max()
            
            # Store values in dataframe at the original index
            idx = self.df.index[i]  # Get the actual index value
            self.df.at[idx, 'best_low'] = low
            self.df.at[idx, 'best_high'] = high
            
            # Calculate Fast %K with check for division by zero
            if high != low:  # Avoid division by zero
                self.df.at[idx, 'fast_k'] = 100 * ((self.df.iloc[i]['close'] - low) / (high - low))
            else:
                self.df.at[idx, 'fast_k'] = 50  # Default to middle value when high = low
        
        # Calculate derivatives using the DataFrame's original index
        self.df['fast_d'] = self.df['fast_k'].rolling(3).mean().round(2)
        self.df['slow_k'] = self.df['fast_d']
        self.df['slow_d'] = self.df['slow_k'].rolling(3).mean().round(2)
        
    # =============================================================================
    # Section: Time-based Features
    # =============================================================================
    def _time_of_day(self):
        pass
    
    def _time_of_week(self):
        pass
    
    def _time_of_month(self):
        pass
    
    def _day_of_week(self):
        # Unable to access date when it is the index, so thought this would be an okay solution
        self.df['date'] = self.df.index

        self.df["day_of_week"] = self.df["date"].dt.dayofweek
        self.df["sin_day_of_week"] = np.sin(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["cos_day_of_week"] = np.cos(2 * np.pi * self.df["day_of_week"] / 7)
        # Removing date duplicate again
        self.df = self.df.drop('date', axis=1)
        
    def _seasonal_decompose(self, period = 160):
        """Not working properly"""
        try:
            self.df["sd"] = seasonal_decompose(self.df["date"], model='additive', period=period, extrapolate_trend=1)
            self.df["stl"] = STL(self.df["date"], period=period)
            self.df["stl"] = MSTL(self.df["date"], periods=[24, 160])
        except Exception as e:
            logging.warning(f"Error calculating seasonal decompose: {e}")
    
    # =============================================================================
    # Section: Market Regime Features
    # =============================================================================
    def _volatility_regimes(self):
        """How do I implement this?"""
    
    # =============================================================================
    # Section: Feature Transformations
    # =============================================================================
    def _log_returns(self):
        """Logarithmic returns: measure the percentage change in the price of an asset over a specific period."""
        self.df['log_return'] = np.log( self.df['close'] / self.df['close'].shift(1) )
        
    def _normalize_features(self, columns=None, window=20, exclude=None):  #['open', 'high', 'low', 'close']
        """
        Z-score normalize selected features in the DataFrame.
        
        Parameters:
        -----------
        columns : list or None
            List of columns to normalize. If None, normalizes all numeric columns.
        window : int
            The rolling window to use for calculating mean and standard deviation.
        exclude : list or None
            List of columns to exclude from normalization.
            
        Returns:
        --------
        None. Adds new columns to self.df with '_zscore' suffix.
        """
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = self.df.select_dtypes(include='number').columns.tolist()
        
        # Exclude specified columns
        if exclude is not None:
            columns = [col for col in columns if col not in exclude]
        
        # Skip columns that already have a z-score version
        columns = [col for col in columns if f"{col}_zscore" not in self.df.columns]
        
        # Apply z-score normalization to each column
        for col in columns:
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
            
            # Add to dataframe
            self.df[f'{col}_zscore'] = z_score

    def _fast_fourier_transforms(self):
        "For extracting cyclical components."
        # self.df["fft"] = fft(self.df["close"])
        
        # Number of samplepoints
        N = len(self.df)
        
        # Sample spacing
        T = 1.0 / 800.0  # ?
        x = np.linspace(0.0, N*T, N)
        y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
        self.df["fft"] = fft(y)  # is this how it's used?? - weird reuslts, eg. 1.3549970-0.0000000j
        