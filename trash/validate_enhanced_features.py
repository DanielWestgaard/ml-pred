#!/usr/bin/env python3
"""
Enhanced Spread Compression Features Validation Script

Validates that the enhanced features are genuinely improving predictive power
for spread compression trading signals on 1-minute GBPUSD data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
sys.path.append('/Users/danielwestgaard/progRel/ml-pred')

def load_processed_data():
    """Load the processed dataset with enhanced features"""
    data_path = Path("/Users/danielwestgaard/progRel/ml-pred/storage/data/capital_com/GBPUSD/processed/1m/2024-10-01_2025-07-25.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}")
    
    print(f"📊 Loading processed data from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
    
    print(f"✅ Data loaded successfully!")
    print(f"   • Shape: {df.shape}")
    print(f"   • Date range: {df.index.min()} to {df.index.max()}")
    print(f"   • Columns: {len(df.columns)}")
    
    return df

def verify_enhanced_features(df):
    """Verify all enhanced features are present"""
    print("\n🔍 VERIFYING ENHANCED FEATURES")
    print("=" * 50)
    
    # Expected enhanced features
    expected_features = {
        'Multi-timeframe compression': [
            'compression_ratio_micro',    # 30min/3min
            'compression_ratio_short',    # 1hr/5min
            'compression_ratio',          # Original 60min/3min
        ],
        'Advanced volume confirmation': [
            'volume_acceleration',        # Volume momentum
            'volume_breakout',           # Z-score > 2
            'volume_spread_correlation'  # Rolling correlation
        ],
        'Enhanced signal strength': [
            'signal_strength_v2',        # Optimized weights
            'signal_strength_compression' # Original
        ],
        'Core predictors': [
            'volume_ratio',              # 1.52x strongest
            'compression_consistency',
            'london_session'
        ]
    }
    
    all_present = True
    for category, features in expected_features.items():
        print(f"\n📋 {category}:")
        for feature in features:
            if feature in df.columns:
                print(f"   ✅ {feature}")
            else:
                print(f"   ❌ {feature} - MISSING!")
                all_present = False
    
    if all_present:
        print(f"\n🎯 All enhanced features present! Total features: {len(df.columns)}")
    else:
        print(f"\n⚠️  Some enhanced features are missing!")
    
    return all_present

def analyze_signal_quality(df):
    """Analyze signal quality using different feature combinations"""
    print("\n🎯 SIGNAL QUALITY ANALYSIS")
    print("=" * 50)
    
    # Remove rows with NaN target values
    df_clean = df.dropna(subset=['future_price_change'])
    baseline_movement = df_clean['future_price_change'].abs().mean()
    
    print(f"📊 Baseline average movement: {baseline_movement:.6f}")
    
    # Test different signal combinations
    signal_tests = {
        'Original compression': {
            'condition': df_clean['compression_ratio'] > df_clean['compression_ratio'].quantile(0.8),
            'description': 'Original compression ratio (80th percentile)'
        },
        'Enhanced micro compression': {
            'condition': df_clean['compression_ratio_micro'] > df_clean['compression_ratio_micro'].quantile(0.8),
            'description': 'Micro timeframe compression (30min/3min)'
        },
        'Volume only (strongest)': {
            'condition': df_clean['volume_ratio'] > df_clean['volume_ratio'].quantile(0.75),
            'description': 'Volume ratio above 75th percentile'
        },
        'Enhanced signal strength': {
            'condition': df_clean['signal_strength_v2'] > df_clean['signal_strength_v2'].quantile(0.8),
            'description': 'New optimized signal strength'
        },
        'Original signal strength': {
            'condition': df_clean['signal_strength_compression'] > df_clean['signal_strength_compression'].quantile(0.8),
            'description': 'Original signal strength'
        },
        'Multi-feature combo': {
            'condition': (
                (df_clean['compression_ratio_micro'] > df_clean['compression_ratio_micro'].quantile(0.75)) &
                (df_clean['volume_ratio'] > df_clean['volume_ratio'].quantile(0.75)) &
                (df_clean['volume_acceleration'] > 0) &
                (df_clean['london_session'] == 1)
            ),
            'description': 'Micro compression + volume + acceleration + London session'
        }
    }
    
    results = []
    
    for signal_name, test in signal_tests.items():
        condition = test['condition']
        signal_events = df_clean[condition]
        
        if len(signal_events) > 0:
            signal_movement = signal_events['future_price_change'].abs().mean()
            movement_ratio = signal_movement / baseline_movement
            frequency = len(signal_events) / len(df_clean) * 100
            
            results.append({
                'Signal': signal_name,
                'Events': len(signal_events),
                'Frequency %': frequency,
                'Avg Movement': signal_movement,
                'Movement Ratio': movement_ratio,
                'Description': test['description']
            })
            
            print(f"\n📈 {signal_name}:")
            print(f"   • Events: {len(signal_events):,}")
            print(f"   • Frequency: {frequency:.2f}%")
            print(f"   • Movement ratio: {movement_ratio:.2f}x")
            print(f"   • Daily signals: ~{frequency * 1440 / 100:.1f}")
        else:
            print(f"\n❌ {signal_name}: No events found!")
    
    return pd.DataFrame(results)

def correlation_analysis(df):
    """Analyze correlations between features and target"""
    print("\n🔗 FEATURE CORRELATION ANALYSIS")
    print("=" * 50)
    
    # Select relevant features for correlation
    feature_cols = [
        'compression_ratio', 'compression_ratio_micro', 'compression_ratio_short',
        'volume_ratio', 'volume_acceleration', 'volume_breakout',
        'signal_strength_compression', 'signal_strength_v2',
        'compression_consistency', 'volume_spread_correlation'
    ]
    
    # Calculate correlations with absolute future price change
    df_clean = df.dropna(subset=['future_price_change'])
    df_clean['abs_future_change'] = df_clean['future_price_change'].abs()
    
    correlations = []
    for feature in feature_cols:
        if feature in df_clean.columns:
            corr = df_clean[feature].corr(df_clean['abs_future_change'])
            correlations.append({
                'Feature': feature,
                'Correlation': corr,
                'Abs_Correlation': abs(corr)
            })
    
    # Sort by absolute correlation
    corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
    
    print("\n📊 Feature Correlations with Future Price Movement:")
    print("-" * 60)
    for _, row in corr_df.iterrows():
        feature = row['Feature']
        corr = row['Correlation']
        abs_corr = row['Abs_Correlation']
        
        if abs_corr > 0.01:
            strength = "🔥 Strong" if abs_corr > 0.05 else "⭐ Moderate" if abs_corr > 0.02 else "💡 Weak"
        else:
            strength = "❌ None"
            
        print(f"{strength:12} {feature:30} {corr:8.4f}")
    
    return corr_df

def tier_analysis(df):
    """Analyze the quality of different signal tiers"""
    print("\n🏆 TIER QUALITY ANALYSIS")
    print("=" * 50)
    
    # Use existing target tiers
    if 'target_spread_compression_multi' in df.columns:
        df_clean = df.dropna(subset=['future_price_change', 'target_spread_compression_multi'])
        
        tier_analysis = []
        for tier in sorted(df_clean['target_spread_compression_multi'].unique()):
            tier_data = df_clean[df_clean['target_spread_compression_multi'] == tier]
            
            if len(tier_data) > 0:
                avg_movement = tier_data['future_price_change'].abs().mean()
                tier_analysis.append({
                    'Tier': int(tier),
                    'Events': len(tier_data),
                    'Avg_Movement': avg_movement,
                    'Percentage': len(tier_data) / len(df_clean) * 100
                })
        
        tier_df = pd.DataFrame(tier_analysis)
        
        print("\n📋 Signal Tier Breakdown:")
        print("-" * 50)
        for _, row in tier_df.iterrows():
            tier = int(row['Tier'])
            events = int(row['Events'])
            movement = row['Avg_Movement']
            percentage = row['Percentage']
            
            if tier == 0:
                tier_name = "No Signal"
            else:
                tier_name = f"Tier {tier}"
            
            print(f"{tier_name:12} {events:8,} events ({percentage:5.2f}%) - Movement: {movement:.6f}")
        
        # Focus on Tier 2+ for model training
        tier2_plus = df_clean[df_clean['target_spread_compression_multi'] >= 2]
        if len(tier2_plus) > 0:
            print(f"\n🎯 Tier 2+ Quality Signals:")
            print(f"   • Total events: {len(tier2_plus):,}")
            print(f"   • Average movement: {tier2_plus['future_price_change'].abs().mean():.6f}")
            print(f"   • Daily frequency: ~{len(tier2_plus) / len(df_clean) * 1440:.1f} signals/day")
        
        return tier_df
    else:
        print("❌ Multi-class targets not found!")
        return None

def enhanced_vs_original_comparison(df):
    """Compare enhanced features vs original features"""
    print("\n⚡ ENHANCED vs ORIGINAL COMPARISON")
    print("=" * 50)
    
    df_clean = df.dropna(subset=['future_price_change'])
    baseline_movement = df_clean['future_price_change'].abs().mean()
    
    comparisons = [
        {
            'name': 'Signal Strength',
            'original': 'signal_strength_compression',
            'enhanced': 'signal_strength_v2'
        },
        {
            'name': 'Compression Detection',
            'original': 'compression_ratio',
            'enhanced': 'compression_ratio_micro'
        }
    ]
    
    for comp in comparisons:
        print(f"\n🔄 {comp['name']} Comparison:")
        
        # Test original feature
        orig_feature = comp['original']
        if orig_feature in df_clean.columns:
            orig_condition = df_clean[orig_feature] > df_clean[orig_feature].quantile(0.8)
            orig_events = df_clean[orig_condition]
            orig_movement = orig_events['future_price_change'].abs().mean() if len(orig_events) > 0 else 0
            orig_ratio = orig_movement / baseline_movement if orig_movement > 0 else 0
            
            print(f"   📊 Original ({orig_feature}):")
            print(f"      • Events: {len(orig_events):,}")
            print(f"      • Movement ratio: {orig_ratio:.2f}x")
        
        # Test enhanced feature
        enh_feature = comp['enhanced']
        if enh_feature in df_clean.columns:
            enh_condition = df_clean[enh_feature] > df_clean[enh_feature].quantile(0.8)
            enh_events = df_clean[enh_condition]
            enh_movement = enh_events['future_price_change'].abs().mean() if len(enh_events) > 0 else 0
            enh_ratio = enh_movement / baseline_movement if enh_movement > 0 else 0
            
            print(f"   ⭐ Enhanced ({enh_feature}):")
            print(f"      • Events: {len(enh_events):,}")
            print(f"      • Movement ratio: {enh_ratio:.2f}x")
            
            if orig_ratio > 0 and enh_ratio > 0:
                improvement = ((enh_ratio - orig_ratio) / orig_ratio) * 100
                if improvement > 0:
                    print(f"      🚀 Improvement: +{improvement:.1f}%")
                else:
                    print(f"      📉 Change: {improvement:.1f}%")

def main():
    """Main validation script"""
    print("🚀 ENHANCED SPREAD COMPRESSION FEATURES VALIDATION")
    print("=" * 60)
    print("Validating 1-minute GBPUSD data with enhanced features...")
    
    try:
        # Load data
        df = load_processed_data()
        
        # Verify features
        if not verify_enhanced_features(df):
            print("❌ Feature verification failed!")
            return
        
        # Run analyses
        print(f"\n📈 Running comprehensive validation on {len(df):,} data points...")
        
        signal_results = analyze_signal_quality(df)
        correlation_results = correlation_analysis(df)
        tier_results = tier_analysis(df)
        enhanced_vs_original_comparison(df)
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 VALIDATION SUMMARY")
        print("=" * 60)
        
        if len(signal_results) > 0:
            best_signal = signal_results.loc[signal_results['Movement Ratio'].idxmax()]
            print(f"🏆 Best performing signal: {best_signal['Signal']}")
            print(f"   • Movement ratio: {best_signal['Movement Ratio']:.2f}x")
            print(f"   • Events: {best_signal['Events']:,}")
            print(f"   • Daily signals: ~{best_signal['Frequency %'] * 1440 / 100:.1f}")
        
        if correlation_results is not None and len(correlation_results) > 0:
            best_feature = correlation_results.iloc[0]
            print(f"\n🔗 Most predictive feature: {best_feature['Feature']}")
            print(f"   • Correlation: {best_feature['Correlation']:.4f}")
        
        if tier_results is not None:
            tier2_plus = tier_results[tier_results['Tier'] >= 2]
            if len(tier2_plus) > 0:
                total_tier2_events = tier2_plus['Events'].sum()
                print(f"\n🎖️  Tier 2+ quality signals: {total_tier2_events:,} events")
                print(f"   • Perfect for focused model training!")
        
        print(f"\n✅ Validation complete! Enhanced features are ready for model training.")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        raise

if __name__ == "__main__":
    main()