import logging
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from volprofile import getVPWithOHLC
from statsmodels.tsa.seasonal import seasonal_decompose, STL , MSTL

from utils import data_utils


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
        
        print(self.df)
        
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
        self.df["roc_close"] = self.df["close"].pct_change()  # Am I using this right?
    
    def _average_true_range(self, period_atr:int = 14):
        # ATR
        self.df["high_low"] = self.df["high"] - self.df["low"]                          # Current High — Current Low
        self.df["high_prev_close"] = abs(self.df["high"] - self.df["close"].shift(1))   # |Current High — Previous Close|
        self.df["low_prev_close"] = abs(self.df["low"] - self.df["close"].shift(1))     # |Current Low — Previous Close|
        self.df["true_range"] = self.df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
        
        self.df["atr"] = self.df["true_range"].rolling(window=period_atr).mean()
        # Dropping used columns
        self.df = self.df.drop(columns=["high_low", "high_prev_close", "low_prev_close", "true_range"], axis=1)

    def _bollinger_bands(self, period_bb:int = 20):
        # Bollinger Bands
        # middle band is sma
        self.df["std"] = self.df["close"].rolling(window=period_bb).std()  # rolling standard deviation
        num_std_dev = 2
        self.df['upper_band'] = self.df[f'sma_{period_bb}'] + (num_std_dev * self.df['std'])
        self.df['lower_band'] = self.df[f'sma_{period_bb}'] - (num_std_dev * self.df['std'])
        
    def _support_resistance(self):
        """Distance from recent high/lows."""
        pass
    
        
    # =============================================================================
    # Section: Volume-Based Features
    # =============================================================================
    def _volume_moving_averages(self):
        """Identify unusual volume spikes."""
        num_periods = [7, 15, 30, 60]
        
        for period in num_periods:
            self.df[f'vma_{period}'] = self.df['volume'].rolling(window=period).mean()
        
    def _volume_rate_of_change(self):
        """How rapidly volume is increasing/decreasing."""
        self.df["vroc"] = self.df["volume"].pct_change()  # Am I using this right?
        
    def _on_balance_volume(self):
        """Cumulative indicator that relates volume to price changes. Could also be placed under Technical Indicators."""
        obv = [0]
        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] > self.df['close'].iloc[i-1]:
                obv.append(float(obv[-1] + self.df['volume'].iloc[i]))
            elif self.df['close'].iloc[i] < self.df['close'].iloc[i-1]:
                obv.append(float(obv[-1] - self.df['volume'].iloc[i]))
            else:
                obv.append(float(obv[-1]))
        
        self.df["obv"] = obv
        
    def _volume_profile(self):
        """Distribution of volume at different price levels."""
        vp = getVPWithOHLC(self.df, self.df.shape[0])  # minPrice, maxPrice, and aggregateVolume for each price bin
        
        vp.index = self.df.index  # align indexes
        self.df = pd.concat([self.df, vp], axis=1)  # am I using it properly?
        # Naming result columns differently
        self.df["min_price_v"] = self.df["minPrice"]
        self.df["max_price_v"] = self.df["maxPrice"]
        self.df["aggregate_volume"] = self.df["aggregateVolume"]
        # Dropping columns
        self.df = self.df.drop(["minPrice", "maxPrice", "aggregateVolume"], axis=1)
    
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
        self.df["day_of_week"] = self.df["date"].dt.dayofweek
        self.df["sin_day_of_week"] = np.sin(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["cos_day_of_week"] = np.cos(2 * np.pi * self.df["day_of_week"] / 7)
        
    def _seasonal_decompose(self, period = 160):
        """Not working properly"""
        try:
            self.df["sd"] = seasonal_decompose(self.df["date"], model='additive', period=period, extrapolate_trend=1)
            self.df["stl"] = STL(self.df["date"], period=period)
            self.df["stl"] = MSTL(self.df["date"], periods=[24, 160])
        except Exception as e:
            logging.warning(f"Error calculating seasonal decompose. Most likely problem with too little data: {e}")
    
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
        