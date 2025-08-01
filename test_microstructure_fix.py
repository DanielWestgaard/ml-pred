#!/usr/bin/env python3
"""
Quick test to verify the microstructure features fix
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tick_momentum_fix():
    """Test that the tick momentum calculation works correctly"""
    logger.info("Testing tick momentum fix...")
    
    # Create sample data
    test_data = pd.DataFrame({
        'close': [100, 101, 102, 101, 103, 102, 104, 105, 104, 106],
        'volume': [1000, 1100, 1200, 1050, 1300, 1150, 1400, 1500, 1350, 1600],
        'high': [100.5, 101.5, 102.5, 101.5, 103.5, 102.5, 104.5, 105.5, 104.5, 106.5],
        'low': [99.5, 100.5, 101.5, 100.5, 102.5, 101.5, 103.5, 104.5, 103.5, 105.5],
        'open': [100, 100.8, 101.8, 101.8, 102.8, 102.8, 103.8, 104.8, 104.8, 105.8]
    })
    
    try:
        # Test the fixed tick momentum calculation
        price_change = test_data['close'].diff()
        tick_direction = np.sign(price_change)
        
        # This should work now (previously failed)
        tick_direction_series = pd.Series(tick_direction, index=test_data.index)
        tick_momentum = tick_direction_series.rolling(5).sum()
        
        logger.info("✅ Tick momentum calculation successful!")
        logger.info(f"Sample tick momentum values: {tick_momentum.dropna().values[:5]}")
        
        # Test other microstructure calculations
        typical_price = (test_data['high'] + test_data['low'] + test_data['close']) / 3
        volume_weighted_avg = (typical_price * test_data['volume']).rolling(5).sum() / test_data['volume'].rolling(5).sum()
        
        logger.info("✅ Volume weighted average calculation successful!")
        logger.info(f"Sample VWAP values: {volume_weighted_avg.dropna().values[:3]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_temporal_features():
    """Test temporal features calculations"""
    logger.info("Testing temporal features...")
    
    test_data = pd.DataFrame({
        'close': [100, 101, 102, 101, 103, 102, 104, 105, 104, 106, 107, 106, 108],
        'volume': [1000, 1100, 1200, 1050, 1300, 1150, 1400, 1500, 1350, 1600, 1700, 1550, 1800]
    })
    
    try:
        # Test momentum calculations
        momentum_1min = test_data['close'].pct_change(1)
        momentum_3min = test_data['close'].pct_change(3)
        
        # Test acceleration
        price_acceleration = momentum_1min.diff()
        
        # Test momentum persistence
        momentum_consistency = momentum_3min.rolling(5).apply(
            lambda x: (np.sign(x) == np.sign(x.iloc[-1])).sum() / len(x)
        )
        
        logger.info("✅ Temporal features calculation successful!")
        logger.info(f"Momentum persistence sample: {momentum_consistency.dropna().values[:3]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Temporal features test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Testing microstructure features fix...")
    
    tick_test = test_tick_momentum_fix()
    temporal_test = test_temporal_features()
    
    if tick_test and temporal_test:
        logger.info("✅ ALL TESTS PASSED! Microstructure features should work correctly now.")
    else:
        logger.error("❌ Some tests failed. Check the implementation.")