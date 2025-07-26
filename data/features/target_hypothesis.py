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
            # Required features check
            required_features = [
                'compression_ratio', 'spread_volatility_ewm', 'compression_consistency',
                'volume_ratio', 'london_session', 'signal_strength_compression'
            ]
            
            missing_features = [f for f in required_features if f not in self.df.columns]
            if missing_features:
                logger.error(f"Missing required features: {missing_features}")
                return
            
            # Future price movement (next 15 periods = 75 minutes for 5m data)
            self.df['future_price_change'] = (
                self.df['close'].shift(-15) - self.df['close']
            ) / self.df['close']
            
            # STEP 1: Analyze actual data distributions first
            logger.info("=== DATA DISTRIBUTION ANALYSIS ===")
            logger.info(f"Compression ratio - Mean: {self.df['compression_ratio'].mean():.4f}, "
                    f"Std: {self.df['compression_ratio'].std():.4f}, "
                    f"95th percentile: {self.df['compression_ratio'].quantile(0.95):.4f}")
            
            logger.info(f"Compression consistency - Mean: {self.df['compression_consistency'].mean():.4f}, "
                    f"95th percentile: {self.df['compression_consistency'].quantile(0.95):.4f}")
            
            logger.info(f"Volume ratio - Mean: {self.df['volume_ratio'].mean():.4f}, "
                    f"95th percentile: {self.df['volume_ratio'].quantile(0.95):.4f}")
            
            logger.info(f"Future price change - Std: {self.df['future_price_change'].std():.6f}, "
                    f"95th percentile: {abs(self.df['future_price_change']).quantile(0.95):.6f}")
            
            # STEP 2: Use adaptive thresholds based on actual data
            compression_threshold = self.df['compression_ratio'].quantile(0.80)  # Top 20%
            consistency_threshold = self.df['compression_consistency'].quantile(0.70)  # Top 30%
            volume_threshold = max(1.1, self.df['volume_ratio'].quantile(0.75))  # Top 25% but at least 1.1
            movement_threshold = abs(self.df['future_price_change']).quantile(0.75)  # Top 25% movements
            
            logger.info(f"Adaptive thresholds - Compression: {compression_threshold:.4f}, "
                    f"Consistency: {consistency_threshold:.4f}, "
                    f"Volume: {volume_threshold:.4f}, "
                    f"Movement: {movement_threshold:.6f}")
            
            # STEP 3: Progressive signal strength (instead of hard AND conditions)
            
            # Basic compression signal
            compression_signal = self.df['compression_ratio'] > compression_threshold
            
            # Add consistency bonus
            compression_with_consistency = (
                compression_signal & 
                (self.df['compression_consistency'] > consistency_threshold)
            )
            
            # Add volume confirmation
            strong_compression_signal = (
                compression_with_consistency & 
                (self.df['volume_ratio'] > volume_threshold)
            )
            
            # Future significant movement
            significant_movement = (
                abs(self.df['future_price_change']) > movement_threshold
            )
            
            # STEP 4: Create tiered targets
            
            # Tier 1: Basic compression + movement
            basic_signal = compression_signal & significant_movement
            
            # Tier 2: Compression + consistency + movement  
            medium_signal = compression_with_consistency & significant_movement
            
            # Tier 3: Full signal (compression + consistency + volume + movement)
            strong_signal = strong_compression_signal & significant_movement
            
            # Tier 4: London session bonus
            london_strong_signal = strong_signal & (self.df['london_session'] == 1)
            
            # STEP 5: Create multi-class target
            target_values = np.where(
                london_strong_signal, 4,  # Best signal
                np.where(
                    strong_signal, 3,     # Strong signal
                    np.where(
                        medium_signal, 2,  # Medium signal
                        np.where(
                            basic_signal, 1,  # Basic signal
                            0  # No signal
                        )
                    )
                )
            )
            
            self.df['target_spread_compression_multi'] = target_values
            
            # Binary version (any signal vs no signal)
            self.df['target_spread_compression_binary'] = (target_values > 0).astype(int)
            
            # STEP 6: Report results
            target_distribution = pd.Series(target_values).value_counts().sort_index()
            logger.info(f"Multi-class target distribution: {target_distribution.to_dict()}")
            logger.info(f"Binary target distribution: {self.df['target_spread_compression_binary'].value_counts().to_dict()}")
            
            # Signal frequency analysis
            total_signals = (target_values > 0).sum()
            signal_frequency = total_signals / len(self.df) * 100
            logger.info(f"Signal frequency: {total_signals}/{len(self.df)} = {signal_frequency:.2f}% of periods")
            
            # Estimate daily frequency (assuming 5-minute bars)
            periods_per_day = 288  # 24 * 60 / 5
            estimated_daily_signals = signal_frequency / 100 * periods_per_day
            logger.info(f"Estimated daily signals: {estimated_daily_signals:.1f}")
            
            # STEP 7: Quick validation of each tier
            for tier in [1, 2, 3, 4]:
                tier_mask = target_values == tier
                if tier_mask.sum() > 0:
                    tier_movement = abs(self.df[tier_mask]['future_price_change']).mean()
                    logger.info(f"Tier {tier} average movement: {tier_movement:.6f} ({tier_mask.sum()} events)")
            
            return True
            
        except Exception as e:
            logger.error(f"Unable to calculate fixed target: {e}")
            return False

    # IMPORTANT FOR FURTHER DEVELOPMENT: Remember to add the prefix 'target_*' for the target column to the hypothesis.
    # This is because the normalization step after generation and this step will not normalize columns that starts with 'target_'.

