import inspect
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from data.processing.base_processor import BaseProcessor
from utils import data_utils
from utils import shared_utils
from utils.logging_utils import get_logger

logger = get_logger()

class TargetHypothesis(BaseProcessor):
    """Change target of dataset according to the hypothesis."""
    def __init__(self, dataset, hypothesis:str):
        """
        Initialize hypothesis to change target column.
        
        Args:
            dataset: Processed data with specific features generated.
            hypothesis: The hypothesis you intend to test. Must match the name of the function you want to target!
        """
        try:
            # Load dataset based on format
            self.df, self.original_df = data_utils.check_and_return_df(dataset)
            
            self.hypothesis = hypothesis
        except Exception as e:
            logger.error(f"Unable to initialize Target Hypothesis: {e}")
    
    def run(self):
        """
        Change target.
        
        returns:
            dataset: with added new target.
        """
        logger.info("---------- Running Hypothesis Targeting ----------")
        
        # Making a list of existing methods in this class
        functions = shared_utils.list_of_functions_in_class(path_to_class='data.features.target_hypothesis.TargetHypothesis')
        
        try:
            successfull = False
            for hypotehesis in functions:
                if hypotehesis == self.hypothesis:  # The one hypothesis user wants to target
                    successfull = shared_utils.safely_execute(self, hypotehesis)
            if not successfull:
                logger.error(f"Could not find {self.hypothesis} in the list of functions: {functions}. Make sure your name matches the function you want to target!")
        except Exception as e:
            logger.error(f"Error trying to find and execute hypothesis method: {e}")
        
        return self.df
    
    def session_based_tod_anomalies(self):
        """
        This target captures when future volatility is abnormally high relative to that specific session's
        typical behavior - directly testing the hypothesis that sessions behave differently.
        """
        try:
            if 'atr' in self.df.columns and 'primary_session' in self.df.columns:
                # Calculate session-specific ATR norms using primary_session feature
                self.df['session_atr_mean'] = self.df.groupby('primary_session')['atr'].transform(
                    lambda x: x.rolling(window=100, min_periods=20).mean()
                )
                self.df['session_atr_std'] = self.df.groupby('primary_session')['atr'].transform(
                    lambda x: x.rolling(window=100, min_periods=20).std()
                )
                
                # Future volatility (next 30 minutes, assuming 1-min bars)
                self.df['future_atr'] = self.df['atr'].shift(-30).rolling(window=30).mean()
                # Target: Future volatility > session mean + 1.5 * session std
                self.df['target_session_vol_anomaly'] = (
                    (self.df['future_atr'] > self.df['session_atr_mean'] + 1.5 * self.df['session_atr_std'])
                ).astype(int)
                
                logger.debug(f"Target column, target_session_vol_anomaly, for hypothesis session_based_tod_anomalies created.")
            else:
                logger.error(f"Target hypothesis column is not possible to calculate based on missing needed features: 'atr' and 'primary_session'")
        except Exception as e:
            logger.error(f"Unable to calculate the target hypothesis column: {e}")
    
    def spread_compression_patterns(self):
        """
        Target for spread compression pattern hypothesis.
        
        This targets moments when spread compression leads to significant price movements,
        particularly during London session trading hours when market liquidity changes.
        """
        try:
            # Required features for this hypothesis
            required_features = [
                'compression_ratio', 'spread_volatility_ewm', 'compression_consistency',
                'volume_ratio', 'london_session', 'signal_strength_compression'
            ]
            
            missing_features = [f for f in required_features if f not in self.df.columns]
            if missing_features:
                logger.error(f"Missing required features for spread_compression_patterns: {missing_features}")
                return
            
            # Future price movement (next 15 periods ahead for 5m data = 75 minutes)
            self.df['future_price_change'] = (
                self.df['close'].shift(-15) - self.df['close']
            ) / self.df['close']
            
            # Define significant movement threshold (0.15% for GBPUSD)
            movement_threshold = 0.0015
            
            # Target conditions:
            # 1. Strong compression ratio (> 0.1 = 10% compression)
            # 2. Consistent compression (> 0.5)
            # 3. Volume confirmation (> 1.2 = 20% above average)
            # 4. During London session hours (bonus weight)
            # 5. Results in significant price movement
            
            compression_signal = (
                (self.df['compression_ratio'] > 0.1) &
                (self.df['compression_consistency'] > 0.5) &
                (self.df['volume_ratio'] > 1.2)
            )
            
            # Enhanced signal during London session
            london_enhanced_signal = (
                compression_signal & 
                (self.df['london_session'] == 1)
            )
            
            # Future significant movement
            significant_movement = (
                abs(self.df['future_price_change']) > movement_threshold
            )
            
            # Target: Compression pattern that leads to significant movement
            # Higher weight for London session occurrences
            target_spread_compression = np.where(
                london_enhanced_signal & significant_movement, 2,  # Strong signal during London
                np.where(
                    compression_signal & significant_movement, 1,  # Regular signal
                    0  # No signal
                )
            ).astype(int)
            target_spread_compression_series = pd.Series(target_spread_compression)
            
            # Create binary version for simpler classification
            self.df['target_spread_compression_binary'] = (
                target_spread_compression_series > 0
            ).astype(int)
            
            logger.info(f"Target columns for spread_compression_patterns hypothesis created.")
            logger.info(f"Target distribution: {target_spread_compression_series.value_counts().to_dict()}")
            
            logger.info("temporarily validating hypothesis suggestions")
            validate_spread_compression_hypothesis(self, self.df)
            
        except Exception as e:
            logger.error(f"Unable to calculate target for spread_compression_patterns hypothesis: {e}")

    # IMPORTANT FOR FURTHER DEVELOPMENT: Remember to add the prefix 'target_*' for the target column to the hypothesis.
    # This is because the normalization step after generation and this step will not normalize columns that starts with 'target_'.


def validate_spread_compression_hypothesis(self, df):
    """
    Quick validation of spread compression -> volatility hypothesis
    before building complex ML models
    """
    
    # 1. BASIC CORRELATION TEST
    logger.info("=== CORRELATION ANALYSIS ===")
    correlation_matrix = df[['compression_ratio', 'future_price_change', 
                        'volume_ratio', 'volatility_short']].corr()
    logger.info(f"Compression ratio vs Future price change: {correlation_matrix.loc['compression_ratio', 'future_price_change']:.3f}")
    
    # 2. COMPRESSION EVENT ANALYSIS
    logger.info("=== COMPRESSION EVENT ANALYSIS ===")
    
    # Define compression events
    compression_events = (df['compression_ratio'] > 0.1) & (df['compression_consistency'] > 0.5)
    
    # Compare future volatility between compression vs non-compression periods
    compression_vol = df[compression_events]['volatility_short'].shift(-1).mean()
    normal_vol = df[~compression_events]['volatility_short'].shift(-1).mean()
    
    logger.info(f"Average volatility after compression events: {compression_vol:.6f}")
    logger.info(f"Average volatility during normal periods: {normal_vol:.6f}")
    logger.info(f"Volatility increase ratio: {compression_vol/normal_vol:.2f}x")
    
    # Statistical significance
    compression_future_vol = df[compression_events]['volatility_short'].shift(-1).dropna()
    normal_future_vol = df[~compression_events]['volatility_short'].shift(-1).dropna()
    
    if len(compression_future_vol) > 10 and len(normal_future_vol) > 10:
        t_stat, p_value = stats.ttest_ind(compression_future_vol, normal_future_vol)
        logger.info(f"T-test p-value: {p_value:.4f} {'(Significant!)' if p_value < 0.05 else '(Not significant)'}")
    
    # 3. LONDON SESSION BONUS VALIDATION
    logger.info("=== LONDON SESSION ANALYSIS ===")
    
    london_compression = compression_events & (df['london_session'] == 1)
    other_compression = compression_events & (df['london_session'] == 0)
    
    if london_compression.sum() > 0 and other_compression.sum() > 0:
        london_success = df[london_compression]['future_price_change'].abs().mean()
        other_success = df[other_compression]['future_price_change'].abs().mean()
        
        logger.info(f"London compression avg movement: {london_success:.6f}")
        logger.info(f"Other session compression avg movement: {other_success:.6f}")
        logger.info(f"London session advantage: {london_success/other_success:.2f}x")
    
    # 4. SIGNAL FREQUENCY ANALYSIS
    logger.info("=== SIGNAL FREQUENCY ===")
    try:
        daily_signals = df.groupby(df['date'])['target_spread_compression_binary'].sum()
        logger.info(f"Average signals per day: {daily_signals.mean():.1f}")
        logger.info(f"Signal frequency: {compression_events.sum()}/{len(df)} = {compression_events.mean()*100:.2f}% of candles")
    except Exception as e:
        logger.warning(f"Unable to perform SIGNAL FREQUENCY ANALYSIS: {e}")
    
    # 5. THRESHOLD SENSITIVITY
    logger.info("=== THRESHOLD SENSITIVITY ===")
    thresholds = [0.05, 0.1, 0.15, 0.2]
    
    for threshold in thresholds:
        events = df['compression_ratio'] > threshold
        if events.sum() > 5:  # Minimum events needed
            future_movements = df[events]['future_price_change'].abs().mean()
            logger.info(f"Threshold {threshold:.2f}: {events.sum()} events, avg movement: {future_movements:.6f}")
    
    # 6. TIMING ANALYSIS
    logger.info("=== TIMING ANALYSIS ===")
    
    # Check if compression leads to movement in different time windows
    time_windows = [5, 10, 15, 30]  # periods ahead
    
    for window in time_windows:
        future_movement = (df['close'].shift(-window) - df['close']).abs() / df['close']
        compression_movement = future_movement[compression_events].mean()
        normal_movement = future_movement[~compression_events].mean()
        
        if not pd.isna(compression_movement) and not pd.isna(normal_movement):
            logger.info(f"{window*5}min window - Compression: {compression_movement:.6f}, Normal: {normal_movement:.6f}, Ratio: {compression_movement/normal_movement:.2f}x")
    
    return {
        'compression_events': compression_events.sum(),
        'total_periods': len(df),
        'signal_frequency': compression_events.mean(),
        'volatility_increase': compression_vol/normal_vol if normal_vol > 0 else None
    }

    # USAGE EXAMPLE:
    # results = validate_spread_compression_hypothesis(your_dataframe)
    # 
    # If this validation shows:
    # - Compression events correlate with higher future volatility (>1.2x increase)
    # - Statistical significance (p < 0.05)
    # - 3-6 signals per day
    # - London session advantage
    # 
    # Then proceed with ML modeling. Otherwise, refine the hypothesis first.
