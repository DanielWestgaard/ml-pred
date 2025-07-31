#!/usr/bin/env python3
"""
Simple Model Diagnostic - Focus on directional accuracy
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from utils.logging_utils import get_logger

logger = get_logger()

def run_simple_diagnostic():
    """Run focused diagnostic on model directional accuracy"""
    
    print("🔬 SIMPLE MODEL DIAGNOSTIC")
    print("=" * 50)
    
    # Load model
    model_folder = Path("storage/models/2025-07-30_19:30:29")
    model_path = model_folder / "random_forest.pkl"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    model = joblib.load(model_path)
    print(f"✅ Model loaded: {type(model).__name__}")
    
    # Load backtesting data (same as used in backtesting)
    data_file = Path("storage/data/capital_com/GBPUSD/processed/1m/2025-03-02_2025-07-01.csv")
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return
    
    data = pd.read_csv(data_file)
    print(f"✅ Data loaded: {len(data):,} rows")
    
    # Check available columns
    print(f"📊 Available columns: {list(data.columns)}")
    
    # Filter to backtesting period (2025-03-10 to 2025-03-20)
    data['timestamp'] = pd.to_datetime(data['date'])
    mask = (data['timestamp'] >= '2025-03-10') & (data['timestamp'] <= '2025-03-20')
    backtest_data = data[mask].copy().reset_index(drop=True)
    
    print(f"✅ Backtesting period data: {len(backtest_data):,} rows")
    
    # Extract the 6 features used by model
    expected_features = ['volume', 'compression_prediction_strength', 'microstructure_health',
                        'volume_ratio', 'spread_baseline', 'volatility_short']
    
    # Check which features are available
    available_features = [f for f in expected_features if f in backtest_data.columns]
    print(f"📊 Available features: {available_features}")
    
    if len(available_features) < 3:
        print("❌ Too few features available for meaningful analysis")
        print("🔍 Let's analyze what we have...")
        
        # Use whatever features are available
        numeric_cols = backtest_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['close', 'open', 'high', 'low']][:6]
    else:
        feature_cols = available_features
    
    missing_features = [f for f in feature_cols if f not in backtest_data.columns]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        return
    
    features = backtest_data[feature_cols].copy()
    print(f"✅ Features extracted: {len(feature_cols)} columns")
    
    # Clean data (remove NaN)
    clean_mask = ~features.isnull().any(axis=1)
    clean_features = features[clean_mask].copy()
    clean_data = backtest_data[clean_mask].copy().reset_index(drop=True)
    
    print(f"✅ Clean data: {len(clean_features):,} rows (dropped {np.sum(~clean_mask):,} NaN)")
    
    if len(clean_features) == 0:
        print("❌ No clean data available for predictions")
        return
    
    # Generate model predictions
    predictions = model.predict(clean_features)
    probabilities = model.predict_proba(clean_features)
    
    print(f"✅ Predictions generated: {len(predictions):,}")
    
    # Analyze prediction distribution
    pred_counts = np.bincount(predictions.astype(int))
    print(f"📊 Prediction distribution:")
    for i, count in enumerate(pred_counts):
        if count > 0:
            print(f"   Class {i}: {count:,} ({count/len(predictions)*100:.1f}%)")
    
    # Analyze confidence distribution
    max_confidence = np.max(probabilities, axis=1)
    print(f"📊 Confidence statistics:")
    print(f"   Min: {np.min(max_confidence):.4f}")
    print(f"   Max: {np.max(max_confidence):.4f}")
    print(f"   Mean: {np.mean(max_confidence):.4f}")
    print(f"   Unique values: {len(np.unique(max_confidence)):,}")
    
    print(f"\n🎯 DIRECTIONAL ACCURACY TEST")
    print("-" * 40)
    
    # Test directional accuracy at different time horizons
    horizons = [1, 3, 5, 10, 15]  # minutes
    
    for horizon in horizons:
        # Calculate actual future returns
        future_prices = clean_data['close'].shift(-horizon)
        current_prices = clean_data['close']
        future_returns = (future_prices / current_prices - 1)
        
        # Convert to directional signals (same logic as your model)
        # Based on your model: 0=HOLD, 1=BUY, 2=SELL
        actual_direction = np.zeros(len(future_returns))
        threshold = 0.0005  # 0.05% threshold for significant moves
        
        actual_direction[future_returns > threshold] = 1   # BUY signal (price goes up)
        actual_direction[future_returns < -threshold] = 2  # SELL signal (price goes down)
        
        # Compare predictions with actual outcomes
        # Only look at trading signals (exclude HOLD predictions)
        trading_mask = (predictions != 0) & (~np.isnan(future_returns))
        
        if np.sum(trading_mask) == 0:
            print(f"   {horizon:2d}min: No trading signals")
            continue
        
        pred_trading = predictions[trading_mask]
        actual_trading = actual_direction[trading_mask]
        
        # Calculate accuracy for different scenarios
        total_signals = len(pred_trading)
        
        # Scenario 1: Exact match (pred BUY -> actual UP, pred SELL -> actual DOWN)
        exact_matches = np.sum(pred_trading == actual_trading)
        exact_accuracy = exact_matches / total_signals if total_signals > 0 else 0
        
        # Scenario 2: Directional correctness (pred BUY -> price went up, pred SELL -> price went down)
        buy_signals = pred_trading == 1
        sell_signals = pred_trading == 2
        
        future_returns_trading = future_returns[trading_mask]
        
        correct_buys = np.sum(buy_signals & (future_returns_trading > 0))
        correct_sells = np.sum(sell_signals & (future_returns_trading < 0))
        total_buys = np.sum(buy_signals)
        total_sells = np.sum(sell_signals)
        
        directional_accuracy = (correct_buys + correct_sells) / total_signals if total_signals > 0 else 0
        
        # Analysis
        avg_return_on_buys = np.mean(future_returns_trading[buy_signals]) if total_buys > 0 else 0
        avg_return_on_sells = np.mean(future_returns_trading[sell_signals]) if total_sells > 0 else 0
        
        print(f"   {horizon:2d}min: {directional_accuracy*100:5.1f}% directional accuracy "
              f"({total_signals:,} signals)")
        
        if total_buys > 0:
            print(f"         BUY signals: {total_buys:,}, avg return: {avg_return_on_buys*10000:+5.1f} bps")
        if total_sells > 0:
            print(f"         SELL signals: {total_sells:,}, avg return: {avg_return_on_sells*10000:+5.1f} bps")
        
        # Diagnosis
        if directional_accuracy < 0.3:
            print(f"         🚨 CRITICAL: Model predictions are strongly INVERSE!")
        elif directional_accuracy < 0.45:
            print(f"         ❌ Model predictions are worse than random")
        elif directional_accuracy < 0.55:
            print(f"         ⚠️  Model predictions are near-random")
        else:
            print(f"         ✅ Model shows predictive power")
    
    print(f"\n💡 SUMMARY DIAGNOSIS:")
    print("-" * 40)
    
    # Generate final diagnosis
    # Focus on 5-minute horizon as it's most relevant for trading
    future_returns_5min = (clean_data['close'].shift(-5) / clean_data['close'] - 1)
    trading_mask_5min = (predictions != 0) & (~np.isnan(future_returns_5min))
    
    if np.sum(trading_mask_5min) > 100:  # Need reasonable sample size
        pred_5min = predictions[trading_mask_5min]
        returns_5min = future_returns_5min[trading_mask_5min]
        
        buy_returns = returns_5min[pred_5min == 1]
        sell_returns = returns_5min[pred_5min == 2]
        
        buy_accuracy = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
        sell_accuracy = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
        overall_accuracy = (np.sum(buy_returns > 0) + np.sum(sell_returns < 0)) / len(pred_5min)
        
        print(f"📊 5-minute analysis ({len(pred_5min):,} signals):")
        print(f"   BUY accuracy: {buy_accuracy*100:.1f}% ({len(buy_returns):,} signals)")
        print(f"   SELL accuracy: {sell_accuracy*100:.1f}% ({len(sell_returns):,} signals)")
        print(f"   Overall accuracy: {overall_accuracy*100:.1f}%")
        
        if overall_accuracy < 0.35:
            print(f"\n🎯 DIAGNOSIS: Model predictions are STRONGLY INVERSE")
            print(f"   💡 SOLUTION: Try inverting your signals (BUY->SELL, SELL->BUY)")
            print(f"   📝 This suggests your model learned the opposite pattern")
        elif overall_accuracy < 0.45:
            print(f"\n🎯 DIAGNOSIS: Model has poor directional accuracy")
            print(f"   💡 SOLUTION: Model needs retraining with better features")
        else:
            print(f"\n🎯 DIAGNOSIS: Model shows some predictive power")
            print(f"   💡 The issue may be in risk management or execution timing")
    else:
        print("❌ Insufficient trading signals for diagnosis")

if __name__ == "__main__":
    run_simple_diagnostic()