#!/usr/bin/env python3
"""
Quick validation script to confirm that target function updates maintain promising results.
Validates the key metrics mentioned in helpClaude.md.
"""

import pandas as pd
import numpy as np
from utils.logging_utils import get_logger

logger = get_logger()

def validate_spread_compression_results():
    """Validate that the updated target function maintains promising results."""
    
    # Load the processed data
    try:
        df = pd.read_csv('storage/data/capital_com/GBPUSD/processed/1m/2024-10-01_2025-07-25.csv')
        logger.info(f"Loaded processed data with {len(df)} rows")
    except FileNotFoundError:
        logger.error("Processed data file not found. Run the hypothesis pipeline first.")
        return False
    
    # Convert date column to timestamp index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Key validation metrics from helpClaude.md
    logger.info("=== VALIDATING KEY METRICS FROM helpClaude.md ===")
    
    # 1. Check signal frequency (should be ~17.1 per day)
    total_signals = df['target_spread_compression_binary'].sum()
    signal_frequency_pct = (total_signals / len(df)) * 100
    periods_per_day = 1440  # 1-minute data = 1440 periods per day
    daily_signals = signal_frequency_pct / 100 * periods_per_day
    
    logger.info(f"Signal frequency: {total_signals}/{len(df)} = {signal_frequency_pct:.2f}% of periods")
    logger.info(f"Estimated daily signals: {daily_signals:.1f}")
    
    # 2. Validate tier distribution (Tier 2+: 721 quality signals)
    tier_distribution = df['target_spread_compression_multi'].value_counts().sort_index()
    tier_2_plus = tier_distribution[tier_distribution.index >= 2].sum()
    
    logger.info(f"Target tier distribution: {tier_distribution.to_dict()}")
    logger.info(f"Tier 2+ signals: {tier_2_plus}")
    
    # 3. Check average movements for each tier
    logger.info("=== TIER MOVEMENT ANALYSIS ===")
    for tier in [1, 2, 3, 4]:
        tier_mask = df['target_spread_compression_multi'] == tier
        if tier_mask.sum() > 0:
            tier_movement = abs(df[tier_mask]['future_price_change']).mean()
            logger.info(f"Tier {tier}: {tier_movement:.6f} avg movement ({tier_mask.sum()} events)")
    
    # 4. Feature completeness check
    required_features = [
        'compression_ratio_micro', 'compression_consistency', 'signal_strength_v2',
        'volume_ratio', 'volume_acceleration', 'volume_breakout', 
        'london_session', 'price_momentum_5'
    ]
    
    logger.info("=== FEATURE COMPLETENESS CHECK ===")
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        logger.error(f"Missing required features: {missing_features}")
        return False
    
    # Check for any NaN values in key features
    for feature in required_features:
        nan_count = df[feature].isna().sum()
        if nan_count > 0:
            logger.warning(f"Feature '{feature}' has {nan_count} NaN values")
    
    logger.info("All required features present in dataset")
    
    # 5. Movement prediction validation (should show predictive power)
    logger.info("=== MOVEMENT PREDICTION VALIDATION ===")
    
    # Calculate movement ratio for different signal strengths
    no_signal_movement = abs(df[df['target_spread_compression_binary'] == 0]['future_price_change']).mean()
    signal_movement = abs(df[df['target_spread_compression_binary'] == 1]['future_price_change']).mean()
    
    movement_ratio = signal_movement / no_signal_movement if no_signal_movement > 0 else 0
    
    logger.info(f"No signal average movement: {no_signal_movement:.6f}")
    logger.info(f"Signal average movement: {signal_movement:.6f}")
    logger.info(f"Movement prediction ratio: {movement_ratio:.2f}x")
    
    # 6. Validate data quality (should be 100% complete)
    total_completeness = (len(df) - df.isna().sum().sum()) / (len(df) * len(df.columns)) * 100
    logger.info(f"Overall data completeness: {total_completeness:.2f}%")
    
    # 7. Success criteria validation
    logger.info("=== SUCCESS CRITERIA VALIDATION ===")
    
    success_criteria = {
        "Daily signals in range 15-20": 15 <= daily_signals <= 20,
        "Movement ratio > 1.2x": movement_ratio > 1.2,
        "Tier 2+ signals > 500": tier_2_plus > 500,
        "Data completeness > 99%": total_completeness > 99,
        "All features present": len(missing_features) == 0
    }
    
    all_passed = True
    for criterion, passed in success_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {criterion}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("🎉 ALL VALIDATION CRITERIA PASSED! Target function update is successful.")
    else:
        logger.error("⚠️  Some validation criteria failed. Review the results above.")
    
    return all_passed

if __name__ == "__main__":
    validate_spread_compression_results()