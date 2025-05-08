import logging
import pandas as pd
import numpy as np
import scipy
from scipy.fft import fft, ifft, fftfreq
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
        self._commodity_channel_index()
        logging.debug("Finished calculating Technical Indicators")
        # Time-based Features
        self._time_of_day()
        self._day_of_week()
        self._time_of_week()
        self._time_of_month()
        self._seasonal_decompose()  # not properly working or handling for different timeframes
        self._stl_decomposition()
        self._mstl_decomposition()
        self._holiday_features()
        logging.debug("Finished calculating Time-based Features")
        # Market Regime Features
        self._volatility_regimes()
        self._trend_strength_indicators()
        self._market_state_indicators()
        # Feature Transformations
        self._log_returns()  # Is this implemented right?
        self._normalize_features()
        self._fast_fourier_transforms()
        
        feature_stats = FeatureStatistics(self.df)
        n_features, feature_list = feature_stats.generated_features()
        logging.info(f"Generated {n_features} features.")
        logging.debug(f"Generated features are: {feature_list}")
        
        # print(self.df)
        
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
    def _relative_strength_index(self, window=14):
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
        self.df[f'rsi_{window}'] = pd.Series(rsi, index=rsi.index)
    
    def _moving_average_convergence_divergence(self):
        """Trend-following momentum indicator."""
        self.df['ema_12'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['ema_26'] = self.df['close'].ewm(span=26, adjust=False).mean()

        # Calculate MACD line (the difference between 12-period EMA and 26-period EMA)
        self.df['macd_line'] = self.df['ema_12'] - self.df['ema_26']
        
        # Calculate the signal line (9-period EMA of MACD)
        self.df['macd_signal'] = self.df['macd_line'].ewm(span=9, adjust=False).mean()
        
        # Calculate MACD histogram
        self.df['macd_histogram'] = self.df['macd_line'] - self.df['macd_signal']
        
        # Drop intermediate EMAs if not needed
        self.df = self.df.drop(["ema_12", "ema_26"], axis=1)
    
    def _average_directional_index(self, lookback=14):
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
        
        # Add to dataframe
        self.df["+di"] = plus_di
        self.df["-di"] = minus_di
        self.df["adx"] = adx
        
    def _stochastic_oscillator(self, k_period=14, d_period=3, slowing=3):
        """Calculate Stochastic Oscillator."""
        # Find the lowest low and highest high for the k_period
        low_min = self.df['low'].rolling(window=k_period).min()
        high_max = self.df['high'].rolling(window=k_period).max()
        
        # Calculate %K (Fast Stochastic)
        # %K = 100 * (current close - lowest low) / (highest high - lowest low)
        k_fast = 100 * (self.df['close'] - low_min) / (high_max - low_min)
        
        # Handle division by zero
        k_fast = k_fast.replace([np.inf, -np.inf], np.nan).fillna(50)
        
        # Apply slowing period to fast %K 
        k = k_fast.rolling(window=slowing).mean() if slowing > 1 else k_fast
        
        # Calculate %D (signal line) - simple moving average of %K
        d = k.rolling(window=d_period).mean()
        
        # Add to dataframe
        self.df['stoch_k'] = k
        self.df['stoch_d'] = d
        
    def _commodity_channel_index(self, period=20):
        """Calculate Commodity Channel Index."""
        # Typical Price
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        # Moving Average of Typical Price
        ma_tp = tp.rolling(window=period).mean()
        
        # Mean Absolute Deviation (MAD)
        mad = abs(tp - ma_tp).rolling(window=period).mean()
        
        # CCI
        cci = (tp - ma_tp) / (0.015 * mad)
        
        self.df['cci'] = cci
    
    def _ichimoku_cloud(self, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
        """Calculate Ichimoku Cloud components."""
        # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past tenkan_period
        tenkan_sen = (self.df['high'].rolling(window=tenkan_period).max() + 
                    self.df['low'].rolling(window=tenkan_period).min()) / 2
        
        # Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past kijun_period
        kijun_sen = (self.df['high'].rolling(window=kijun_period).max() + 
                    self.df['low'].rolling(window=kijun_period).min()) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 displaced forward displacement periods
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past senkou_b_period, displaced forward displacement periods
        senkou_span_b = ((self.df['high'].rolling(window=senkou_b_period).max() + 
                        self.df['low'].rolling(window=senkou_b_period).min()) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span): Current closing price, displaced backwards displacement periods
        chikou_span = self.df['close'].shift(-displacement)
        
        # Add to dataframe
        self.df['ichimoku_tenkan'] = tenkan_sen
        self.df['ichimoku_kijun'] = kijun_sen
        self.df['ichimoku_senkou_a'] = senkou_span_a
        self.df['ichimoku_senkou_b'] = senkou_span_b
        self.df['ichimoku_chikou'] = chikou_span
        
    # =============================================================================
    # Section: Time-based Features
    # =============================================================================
    def _time_of_day(self):
        """Extract time of day features from datetime index."""
        # Extract hour of day
        hour = self.df.index.hour
        
        # Create cyclical features using sine and cosine transformations
        # This preserves the cyclical nature (hour 23 is close to hour 0)
        self.df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # You can also add a categorical feature for specific trading sessions
        # For example: Asian, European, and US sessions
        self.df['session'] = pd.cut(
            hour, 
            bins=[-1, 6, 14, 23],
            labels=['asian_session', 'european_session', 'us_session']
        )
    
    def _time_of_week(self):
        """Extract time of week features from datetime index."""
        # Calculate the hour within the week (0-167)
        day_of_week = self.df.index.dayofweek  # Monday=0, Sunday=6
        hour_of_day = self.df.index.hour
        hour_of_week = day_of_week * 24 + hour_of_day
        
        # Create cyclical features
        self.df['hour_of_week_sin'] = np.sin(2 * np.pi * hour_of_week / 168)  # 168 = 7 days * 24 hours
        self.df['hour_of_week_cos'] = np.cos(2 * np.pi * hour_of_week / 168)
        
        # Flag for weekend
        self.df['is_weekend'] = (day_of_week >= 5).astype(int)
    
    def _time_of_month(self):
        """Extract time of month features from datetime index."""
        # Extract day of month
        day = self.df.index.day
        days_in_month = pd.Series(self.df.index).dt.days_in_month.values
        
        # Normalize day to account for different month lengths
        normalized_day = day / days_in_month
        
        # Create cyclical features
        self.df['month_progress_sin'] = np.sin(2 * np.pi * normalized_day)
        self.df['month_progress_cos'] = np.cos(2 * np.pi * normalized_day)
        
        # Mark start, middle and end of month periods
        self.df['month_period'] = pd.cut(
            normalized_day, 
            bins=[0, 0.33, 0.66, 1],
            labels=['start_of_month', 'mid_month', 'end_of_month']
        )
        
        # Month of year cyclical features
        month = self.df.index.month
        self.df['month_of_year_sin'] = np.sin(2 * np.pi * month / 12)
        self.df['month_of_year_cos'] = np.cos(2 * np.pi * month / 12)
    
    def _day_of_week(self):
        """Extract day of week features from datetime index."""
        day_of_week = self.df.index.dayofweek  # Monday=0, Sunday=6
        
        # Create cyclical features
        self.df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        self.df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
    def _seasonal_decompose(self, period=160):
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
            
    def _stl_decomposition(self, period=160):
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
            
    def _mstl_decomposition(self, periods=[24, 160]):
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
            
    def _holiday_features(self):
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
    def _volatility_regimes(self, lookback_short=20, lookback_long=100, n_regimes=3):
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
        self.df['returns'] = self.df['close'].pct_change()
        self.df['volatility_short'] = self.df['returns'].rolling(window=lookback_short).std() * np.sqrt(252)  # Annualized
        self.df['volatility_long'] = self.df['returns'].rolling(window=lookback_long).std() * np.sqrt(252)
        
        # Volatility ratio (current vol relative to longer-term vol)
        self.df['volatility_ratio'] = self.df['volatility_short'] / self.df['volatility_long']
        
        # Identify volatility regimes using quantiles
        if n_regimes == 2:
            # Simple binary regime (high/low)
            self.df['volatility_regime'] = np.where(
                self.df['volatility_short'] > self.df['volatility_short'].rolling(window=lookback_long).quantile(0.5),
                'high', 'low'
            )
        else:
            # Multi-regime approach (e.g., low/medium/high)
            # Calculate historical percentiles for current volatility
            vol_percentile = self.df['volatility_short'].rolling(window=lookback_long).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
            
            # Assign regimes based on percentiles
            conditions = [
                (vol_percentile < 0.33),
                (vol_percentile >= 0.33) & (vol_percentile < 0.66),
                (vol_percentile >= 0.66)
            ]
            choices = ['low', 'medium', 'high']
            self.df['volatility_regime'] = np.select(conditions, choices, default='medium')
        
        # One-hot encode the regime if needed for ML
        regime_dummies = pd.get_dummies(self.df['volatility_regime'], prefix='vol_regime')
        self.df = pd.concat([self.df, regime_dummies], axis=1)
        
        # Also add a continuous volatility Z-score (how many standard deviations from mean)
        vol_mean = self.df['volatility_short'].rolling(window=lookback_long).mean()
        vol_std = self.df['volatility_short'].rolling(window=lookback_long).std()
        self.df['volatility_zscore'] = (self.df['volatility_short'] - vol_mean) / vol_std
        
        # Volatility acceleration/deceleration
        self.df['volatility_change'] = self.df['volatility_short'].pct_change(5)  # 5-period change rate

    def _trend_strength_indicators(self, lookback=20, adx_threshold=25):
        """
        Create indicators to identify trending vs. ranging markets.
        
        Parameters:
        -----------
        lookback : int
            Window for trend calculations
        adx_threshold : int
            ADX threshold above which market is considered trending (typically 25)
        """
        # 1. Directional Movement Trend Strength (simplified ADX-based)
        # Note: This assumes you've already calculated ADX in your technical indicators
        if 'adx' in self.df.columns:
            # Create trend regime based on ADX
            self.df['adx_trend_regime'] = np.where(self.df['adx'] > adx_threshold, 'trending', 'ranging')
            
            # One-hot encode if needed
            regime_dummies = pd.get_dummies(self.df['adx_trend_regime'], prefix='adx_regime')
            self.df = pd.concat([self.df, regime_dummies], axis=1)
        
        # 2. Price vs Moving Average Trend Indicators
        # Calculate price position relative to moving averages
        for period in [20, 50, 200]:
            # Ensure the SMA exists
            if f'sma_{period}' not in self.df.columns:
                self.df[f'sma_{period}'] = self.df['close'].rolling(window=period).mean()
            
            # Calculate % distance from moving average
            self.df[f'dist_from_sma_{period}'] = (self.df['close'] / self.df[f'sma_{period}'] - 1) * 100
            
            # Direction of moving average (is it sloping up or down)
            self.df[f'sma_{period}_slope'] = self.df[f'sma_{period}'].pct_change(5) * 100  # 5-period slope
        
        # 3. Moving Average Crossovers for trend identification
        self.df['ma_cross_50_200'] = np.where(
            self.df['sma_50'] > self.df['sma_200'], 1, -1  # 1 for bullish, -1 for bearish
        )
        
        # 4. Linear regression indicators for trend strength
        # Fit linear regression on closing prices
        self.df['regression_slope'] = np.nan
        self.df['r_squared'] = np.nan
        
        # Calculate for each window
        for i in range(lookback, len(self.df)):
            # Get window of prices
            y = self.df['close'].iloc[i-lookback:i].values
            x = np.arange(len(y))
            
            # Fit linear regression
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
            
            # Store results
            self.df.iloc[i, self.df.columns.get_loc('regression_slope')] = slope
            self.df.iloc[i, self.df.columns.get_loc('r_squared')] = r_value**2
        
        # Normalize the slope based on price level (for comparison across different price levels)
        self.df['norm_regression_slope'] = self.df['regression_slope'] / self.df['close']
        
        # 5. Combined trend strength indicator
        # Calculate a composite trend strength score (0-100)
        # Higher values indicate stronger trends
        components = []
        
        # Add ADX component (scaled 0-1)
        if 'adx' in self.df.columns:
            components.append(self.df['adx'] / 100)
        
        # Add r-squared component (already 0-1)
        components.append(self.df['r_squared'])
        
        # Add moving average alignment component
        ma_alignment = (
            np.sign(self.df['dist_from_sma_20']) * 
            np.sign(self.df['dist_from_sma_50']) * 
            np.sign(self.df['dist_from_sma_200'])
        )
        # This will be 1 when all are positive, -1 when all negative, 0 when mixed
        ma_aligned = (abs(ma_alignment) == 1).astype(float)
        components.append(ma_aligned)
        
        # Average the components and scale to 0-100
        self.df['trend_strength'] = pd.concat(components, axis=1).mean(axis=1) * 100
        
        # 6. Trend regime classification
        self.df['trend_regime'] = np.where(
            self.df['trend_strength'] > 70, 'strong_trend',
            np.where(self.df['trend_strength'] > 40, 'weak_trend', 'ranging')
        )
        
        # One-hot encode the trend regime
        trend_dummies = pd.get_dummies(self.df['trend_regime'], prefix='trend_regime')
        self.df = pd.concat([self.df, trend_dummies], axis=1)

    def _market_state_indicators(self, window=20):
        """
        Generate overall market state indicators that combine volatility and trend regimes.
        
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
    def _log_returns(self):
        """Logarithmic returns: measure the percentage change in the price of an asset over a specific period."""
        self.df['log_return'] = np.log( self.df['close'] / self.df['close'].shift(1) )

    def _fast_fourier_transforms(self, column='close', n_components=5):
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