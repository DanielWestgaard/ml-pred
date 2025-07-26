#!/usr/bin/env python3
"""
Optimal Feature Selection for Spread Compression Trading

Tests different feature combinations to find the optimal set for 
predicting Tier 2+ spread compression signals.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Add project root to path
import sys
sys.path.append('/Users/danielwestgaard/progRel/ml-pred')

def load_processed_data():
    """Load the processed dataset with optimal features"""
    data_path = Path("/Users/danielwestgaard/progRel/ml-pred/storage/data/capital_com/GBPUSD/processed/1m/2024-10-01_2025-07-25.csv")
    
    print(f"📊 Loading processed data from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
    
    print(f"✅ Data loaded successfully!")
    print(f"   • Shape: {df.shape}")
    print(f"   • Date range: {df.index.min()} to {df.index.max()}")
    
    return df

def analyze_feature_correlations(df):
    """Analyze correlations between features to identify redundancy"""
    print("\n🔗 FEATURE CORRELATION MATRIX ANALYSIS")
    print("=" * 60)
    
    # Focus on the optimal feature set
    optimal_features = [
        'compression_ratio_micro', 'compression_ratio_short', 'compression_consistency',
        'signal_strength_v2', 'volume_ratio', 'volume_acceleration', 'volume_breakout',
        'london_session', 'price_momentum_5'
    ]
    
    # Filter features that exist in the data
    available_features = [f for f in optimal_features if f in df.columns]
    print(f"📋 Analyzing {len(available_features)} optimal features:")
    
    # Create correlation matrix
    df_clean = df[available_features].dropna()
    corr_matrix = df_clean.corr()
    
    # Find highly correlated pairs (> 0.7)
    high_corr_pairs = []
    for i in range(len(available_features)):
        for j in range(i+1, len(available_features)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append({
                    'feature1': available_features[i],
                    'feature2': available_features[j],
                    'correlation': corr_val
                })
    
    if high_corr_pairs:
        print(f"\n⚠️  High correlation pairs (>0.7) - potential redundancy:")
        for pair in high_corr_pairs:
            print(f"   • {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
    else:
        print(f"\n✅ No high correlation pairs found - features are independent!")
    
    # Show correlation with target
    target_corr = df_clean.corrwith(df['future_price_change'].abs()).sort_values(key=abs, ascending=False)
    print(f"\n📊 Feature correlations with |future_price_change|:")
    print("-" * 50)
    for feature, corr in target_corr.items():
        strength = "🔥" if abs(corr) > 0.05 else "⭐" if abs(corr) > 0.02 else "💡" if abs(corr) > 0.01 else "❌"
        print(f"{strength} {feature:30} {corr:8.4f}")
    
    return corr_matrix, target_corr, high_corr_pairs

def test_feature_combinations(df, max_features=6):
    """Test different combinations of features for predictive power"""
    print(f"\n🧪 TESTING FEATURE COMBINATIONS (up to {max_features} features)")
    print("=" * 60)
    
    # Core features based on validation results
    core_features = [
        'compression_consistency',    # Strongest predictor (-0.4169)
        'signal_strength_v2',        # Enhanced signal (1.36x ratio)
        'compression_ratio_micro',   # Most sensitive timeframe
        'volume_ratio',              # 1.40x movement ratio
        'london_session',            # Session timing
        'volume_acceleration',       # Volume momentum
        'compression_ratio_short',   # Medium timeframe
        'price_momentum_5',          # Price context
        'volume_breakout'            # Z-score signals
    ]
    
    # Filter available features
    available_features = [f for f in core_features if f in df.columns]
    print(f"📋 Testing combinations from {len(available_features)} features:")
    for f in available_features:
        print(f"   • {f}")
    
    # Prepare data
    df_clean = df.dropna(subset=['future_price_change'] + available_features)
    target = df_clean['future_price_change'].abs()  # Predict absolute movement
    
    # Focus on Tier 2+ signals for training
    tier2_plus = df_clean['target_spread_compression_multi'] >= 2
    tier2_data = df_clean[tier2_plus]
    
    if len(tier2_data) < 100:
        print(f"⚠️  Only {len(tier2_data)} Tier 2+ samples, using all data")
        X = df_clean[available_features]
        y = target
    else:
        print(f"🎯 Using {len(tier2_data)} Tier 2+ samples for training")
        X = tier2_data[available_features]
        y = tier2_data['future_price_change'].abs()
    
    # Test combinations of different sizes
    results = []
    
    for combo_size in range(3, min(max_features + 1, len(available_features) + 1)):
        print(f"\n🔍 Testing {combo_size}-feature combinations...")
        
        best_combo = None
        best_score = -np.inf
        combo_count = 0
        
        for combo in combinations(available_features, combo_size):
            combo_count += 1
            if combo_count > 50:  # Limit combinations for performance
                break
                
            try:
                # Quick cross-validation test
                X_combo = X[list(combo)]
                
                # Use simple train-test split (last 20% as test)
                split_idx = int(len(X_combo) * 0.8)
                X_train, X_test = X_combo.iloc[:split_idx], X_combo.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                model.fit(X_train_scaled, y_train)
                
                # Predict and score
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Combined score (lower MAE, higher R2)
                score = r2 - mae * 1000  # Scale MAE to match R2 range
                
                if score > best_score:
                    best_score = score
                    best_combo = combo
                
            except Exception as e:
                continue
        
        if best_combo:
            results.append({
                'size': combo_size,
                'features': best_combo,
                'score': best_score,
                'tested_combinations': combo_count
            })
            
            print(f"   🏆 Best {combo_size}-feature combo (score: {best_score:.4f}):")
            for feature in best_combo:
                print(f"      • {feature}")
    
    return results

def validate_incremental_features(df):
    """Test adding features incrementally to find optimal set"""
    print(f"\n📈 INCREMENTAL FEATURE SELECTION")
    print("=" * 60)
    
    # Start with the strongest predictor
    core_features = [
        'compression_consistency',    # Strongest predictor
        'signal_strength_v2',        # Enhanced signal  
        'compression_ratio_micro',   # Most sensitive
        'volume_ratio',              # Strong volume signal
        'london_session',            # Session timing
        'volume_acceleration',       # Volume momentum
        'compression_ratio_short',   # Medium timeframe
        'price_momentum_5',          # Price context
        'volume_breakout'            # Z-score signals
    ]
    
    available_features = [f for f in core_features if f in df.columns]
    
    # Prepare data - focus on Tier 2+ signals
    df_clean = df.dropna(subset=['future_price_change'] + available_features)
    tier2_plus = df_clean['target_spread_compression_multi'] >= 2
    
    if tier2_plus.sum() >= 100:
        data = df_clean[tier2_plus]
        print(f"🎯 Using {len(data)} Tier 2+ samples")
    else:
        data = df_clean
        print(f"📊 Using all {len(data)} samples")
    
    target = data['future_price_change'].abs()
    
    # Incremental selection
    selected_features = []
    remaining_features = available_features.copy()
    scores = []
    
    print(f"\n🔍 Starting incremental selection from {len(available_features)} features...")
    
    while remaining_features and len(selected_features) < 8:
        best_feature = None
        best_score = -np.inf
        
        for feature in remaining_features:
            test_features = selected_features + [feature]
            
            try:
                X = data[test_features]
                
                # Simple train-test split
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
                
                # Scale and train
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                score = r2 - mae * 1000
                
                if score > best_score:
                    best_score = score
                    best_feature = feature
                    
            except Exception:
                continue
        
        if best_feature:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            scores.append(best_score)
            
            print(f"   Step {len(selected_features)}: Added '{best_feature}' (score: {best_score:.4f})")
        else:
            break
    
    return selected_features, scores

def test_signal_quality_by_features(df):
    """Test how different feature sets affect signal quality"""
    print(f"\n🎯 SIGNAL QUALITY BY FEATURE SET")
    print("=" * 60)
    
    feature_sets = {
        'compression_only': ['compression_consistency', 'compression_ratio_micro'],
        'volume_only': ['volume_ratio', 'volume_acceleration', 'volume_breakout'],
        'timing_only': ['london_session', 'price_momentum_5'],
        'compression_volume': ['compression_consistency', 'compression_ratio_micro', 'volume_ratio'],
        'enhanced_signal': ['signal_strength_v2'],
        'full_optimal': [
            'compression_consistency', 'signal_strength_v2', 'compression_ratio_micro',
            'volume_ratio', 'london_session', 'volume_acceleration'
        ]
    }
    
    df_clean = df.dropna(subset=['future_price_change'])
    baseline_movement = df_clean['future_price_change'].abs().mean()
    
    results = []
    
    for set_name, features in feature_sets.items():
        available_features = [f for f in features if f in df.columns]
        
        if not available_features:
            continue
            
        # Create combined signal from features
        if len(available_features) == 1:
            signal = df_clean[available_features[0]]
        else:
            # Normalize and combine features
            feature_data = df_clean[available_features]
            normalized = (feature_data - feature_data.mean()) / feature_data.std()
            signal = normalized.mean(axis=1)
        
        # Test signal at different thresholds
        for percentile in [80, 85, 90, 95]:
            threshold = signal.quantile(percentile / 100)
            signal_events = df_clean[signal > threshold]
            
            if len(signal_events) > 0:
                avg_movement = signal_events['future_price_change'].abs().mean()
                movement_ratio = avg_movement / baseline_movement
                frequency = len(signal_events) / len(df_clean) * 100
                
                results.append({
                    'feature_set': set_name,
                    'percentile': percentile,
                    'events': len(signal_events),
                    'frequency': frequency,
                    'movement_ratio': movement_ratio,
                    'daily_signals': frequency * 1440 / 100
                })
    
    # Show best results
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        best_results = results_df.loc[results_df['movement_ratio'].idxmax()]
        
        print(f"\n🏆 Best performing feature set:")
        print(f"   • Set: {best_results['feature_set']}")
        print(f"   • Percentile: {best_results['percentile']}th")
        print(f"   • Movement ratio: {best_results['movement_ratio']:.2f}x")
        print(f"   • Events: {best_results['events']:,}")
        print(f"   • Daily signals: ~{best_results['daily_signals']:.1f}")
        
        # Show top 5 combinations
        top_5 = results_df.nlargest(5, 'movement_ratio')
        print(f"\n📊 Top 5 feature set performances:")
        print("-" * 70)
        for _, row in top_5.iterrows():
            print(f"{row['feature_set']:18} {row['percentile']:3}th percentile: "
                  f"{row['movement_ratio']:.2f}x ({row['events']:4} events, "
                  f"{row['daily_signals']:4.1f} daily)")
    
    return results_df

def main():
    """Main feature optimization script"""
    print("🚀 OPTIMAL FEATURE SELECTION ANALYSIS")
    print("=" * 60)
    print("Finding the best feature combination for spread compression trading...")
    
    try:
        # Load data
        df = load_processed_data()
        
        # Run analyses
        print(f"\n📊 Running comprehensive feature optimization on {len(df):,} data points...")
        
        # 1. Correlation analysis
        corr_matrix, target_corr, high_corr_pairs = analyze_feature_correlations(df)
        
        # 2. Feature combinations
        combination_results = test_feature_combinations(df, max_features=6)
        
        # 3. Incremental selection
        incremental_features, incremental_scores = validate_incremental_features(df)
        
        # 4. Signal quality by feature set
        signal_quality_results = test_signal_quality_by_features(df)
        
        # Final recommendations
        print("\n" + "=" * 60)
        print("🎯 OPTIMAL FEATURE SELECTION RESULTS")
        print("=" * 60)
        
        print(f"\n🏆 INCREMENTAL SELECTION WINNER:")
        if incremental_features:
            print(f"   Optimal feature set ({len(incremental_features)} features):")
            for i, feature in enumerate(incremental_features, 1):
                score = incremental_scores[i-1] if i <= len(incremental_scores) else "N/A"
                print(f"   {i}. {feature} (score: {score:.4f})")
        
        print(f"\n📊 COMBINATION TESTING RESULTS:")
        if combination_results:
            best_combo = max(combination_results, key=lambda x: x['score'])
            print(f"   Best combination ({best_combo['size']} features, score: {best_combo['score']:.4f}):")
            for feature in best_combo['features']:
                print(f"      • {feature}")
        
        if len(signal_quality_results) > 0:
            best_signal = signal_quality_results.loc[signal_quality_results['movement_ratio'].idxmax()]
            print(f"\n🎯 SIGNAL QUALITY CHAMPION:")
            print(f"   • Feature set: {best_signal['feature_set']}")
            print(f"   • Movement ratio: {best_signal['movement_ratio']:.2f}x")
            print(f"   • Daily signals: ~{best_signal['daily_signals']:.1f}")
        
        # Create final recommendation
        print(f"\n✅ FINAL RECOMMENDATION:")
        print(f"   Based on all tests, optimal feature set for model training:")
        
        if incremental_features and len(incremental_features) >= 4:
            # Use top 5-6 from incremental selection
            final_features = incremental_features[:6]
        else:
            # Fallback to top correlations
            final_features = target_corr.head(6).index.tolist()
        
        for i, feature in enumerate(final_features, 1):
            print(f"   {i}. {feature}")
        
        print(f"\n🚀 Ready for model training with {len(final_features)} optimized features!")
        
    except Exception as e:
        print(f"❌ Feature optimization failed: {e}")
        raise

if __name__ == "__main__":
    main()