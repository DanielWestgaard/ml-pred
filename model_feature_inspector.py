#!/usr/bin/env python3
"""
Model Feature Inspector - Find out exactly what features the model expects
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from utils.logging_utils import get_logger

logger = get_logger()

def inspect_model_features():
    """Inspect model to understand what features it was trained on"""
    
    print("🔍 MODEL FEATURE INSPECTOR")
    print("=" * 50)
    
    # Load model
    model_folder = Path("storage/models/2025-07-30_19:30:29")
    model_path = model_folder / "random_forest.pkl"
    
    model = joblib.load(model_path)
    print(f"✅ Model loaded: {type(model).__name__}")
    
    # Check model features
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        print(f"✅ Model expects {len(expected_features)} features:")
        for i, feature in enumerate(expected_features):
            print(f"   {i+1:2d}. {feature}")
    else:
        print("❌ Model doesn't have feature names stored")
        return None
    
    # Load data to see what's available
    data_file = Path("storage/data/capital_com/GBPUSD/processed/1m/2025-03-02_2025-07-01.csv")
    data = pd.read_csv(data_file)
    
    print(f"\n📊 DATA ANALYSIS")
    print("-" * 30)
    print(f"Available columns ({len(data.columns)}):")
    for i, col in enumerate(data.columns):
        print(f"   {i+1:2d}. {col}")
    
    # Check feature overlap
    available_features = set(data.columns)
    required_features = set(expected_features)
    
    missing = required_features - available_features
    extra = available_features - required_features
    
    print(f"\n🔍 FEATURE ANALYSIS")
    print("-" * 30)
    print(f"Required features: {len(required_features)}")
    print(f"Available features: {len(available_features)}")
    print(f"Missing features: {len(missing)}")
    print(f"Extra features: {len(extra)}")
    
    if missing:
        print(f"\n❌ MISSING FEATURES:")
        for feature in sorted(missing):
            print(f"   - {feature}")
    
    if extra:
        print(f"\n➕ EXTRA FEATURES (not used by model):")
        for feature in sorted(list(extra)[:10]):  # Show first 10
            print(f"   - {feature}")
        if len(extra) > 10:
            print(f"   ... and {len(extra)-10} more")
    
    # Try to create a working feature set
    if not missing:
        print(f"\n✅ ALL REQUIRED FEATURES AVAILABLE!")
        print("Model can make predictions on this data")
        
        # Filter to backtesting period
        data['timestamp'] = pd.to_datetime(data['date'])
        mask = (data['timestamp'] >= '2025-03-10') & (data['timestamp'] <= '2025-03-20')
        backtest_data = data[mask].copy()
        
        # Extract model features
        model_features = backtest_data[expected_features].copy()
        
        print(f"\n🧪 QUICK PREDICTION TEST")
        print("-" * 30)
        print(f"Backtesting period: {len(backtest_data):,} rows")
        
        # Clean data
        clean_features = model_features.dropna()
        print(f"Clean data: {len(clean_features):,} rows")
        
        if len(clean_features) > 0:
            # Make predictions
            predictions = model.predict(clean_features[:1000])  # Test first 1000 rows
            probabilities = model.predict_proba(clean_features[:1000])
            
            print(f"✅ Predictions successful!")
            print(f"Prediction distribution: {np.bincount(predictions)}")
            print(f"Confidence range: {np.min(np.max(probabilities, axis=1)):.4f} - {np.max(np.max(probabilities, axis=1)):.4f}")
            
            return expected_features, backtest_data
        else:
            print("❌ No clean data available")
    else:
        print(f"\n❌ CANNOT MAKE PREDICTIONS - MISSING FEATURES")
        print("The model was trained on different data than what's available")
    
    return expected_features, None

def run_directional_accuracy_test(features, data):
    """Run directional accuracy test with correct features"""
    
    if data is None:
        print("❌ No data available for testing")
        return
    
    print(f"\n🎯 DIRECTIONAL ACCURACY TEST")
    print("=" * 50)
    
    # Load model
    model_path = Path("storage/models/2025-07-30_19:30:29/random_forest.pkl")
    model = joblib.load(model_path)
    
    # Prepare features
    model_features = data[features].copy()
    clean_features = model_features.dropna()
    clean_data = data.loc[clean_features.index].copy()
    
    print(f"📊 Clean data: {len(clean_features):,} rows")
    
    # Make predictions
    predictions = model.predict(clean_features)
    probabilities = model.predict_proba(clean_features)
    
    print(f"✅ Predictions generated: {len(predictions):,}")
    
    # Analyze prediction distribution  
    pred_counts = np.bincount(predictions.astype(int))
    print(f"📊 Prediction distribution:")
    for i, count in enumerate(pred_counts):
        if count > 0:
            print(f"   Class {i}: {count:,} ({count/len(predictions)*100:.1f}%)")
    
    # Test directional accuracy
    horizons = [1, 3, 5, 10]
    
    print(f"\n🎯 DIRECTIONAL ACCURACY ANALYSIS")
    print("-" * 40)
    
    for horizon in horizons:
        # Calculate future returns
        future_prices = clean_data['close'].shift(-horizon)
        current_prices = clean_data['close']
        future_returns = (future_prices / current_prices - 1)
        
        # Only analyze trading signals (not HOLD)
        trading_mask = (predictions != 0) & (~np.isnan(future_returns))
        
        if np.sum(trading_mask) == 0:
            print(f"   {horizon:2d}min: No trading signals")
            continue
        
        pred_trading = predictions[trading_mask]
        returns_trading = future_returns[trading_mask]
        
        # Directional accuracy
        buy_signals = pred_trading == 1
        sell_signals = pred_trading == 2
        
        correct_buys = np.sum(buy_signals & (returns_trading > 0))
        correct_sells = np.sum(sell_signals & (returns_trading < 0))
        
        total_signals = len(pred_trading)
        directional_accuracy = (correct_buys + correct_sells) / total_signals
        
        # Return analysis
        buy_returns = returns_trading[buy_signals]
        sell_returns = returns_trading[sell_signals]
        
        avg_buy_return = np.mean(buy_returns) if len(buy_returns) > 0 else 0
        avg_sell_return = np.mean(sell_returns) if len(sell_returns) > 0 else 0
        
        print(f"   {horizon:2d}min: {directional_accuracy*100:5.1f}% accuracy ({total_signals:,} signals)")
        
        if len(buy_returns) > 0:
            print(f"         BUY: {len(buy_returns):,} signals, avg return: {avg_buy_return*10000:+5.1f} bps")
        if len(sell_returns) > 0:
            print(f"         SELL: {len(sell_returns):,} signals, avg return: {avg_sell_return*10000:+5.1f} bps")
        
        # Diagnosis
        if directional_accuracy < 0.35:
            print(f"         🚨 INVERSE: Model predictions are wrong direction!")
        elif directional_accuracy < 0.45:
            print(f"         ❌ POOR: Worse than random")
        elif directional_accuracy < 0.55:
            print(f"         ⚠️  WEAK: Near random performance")
        else:
            print(f"         ✅ GOOD: Shows predictive power")
    
    # Generate final diagnosis
    print(f"\n💡 DIAGNOSIS")
    print("-" * 20)
    
    # Focus on 5-minute results
    future_returns_5min = (clean_data['close'].shift(-5) / clean_data['close'] - 1)
    trading_mask_5min = (predictions != 0) & (~np.isnan(future_returns_5min))
    
    if np.sum(trading_mask_5min) > 50:
        pred_5min = predictions[trading_mask_5min]
        returns_5min = future_returns_5min[trading_mask_5min]
        
        overall_accuracy = (np.sum((pred_5min == 1) & (returns_5min > 0)) + 
                          np.sum((pred_5min == 2) & (returns_5min < 0))) / len(pred_5min)
        
        if overall_accuracy < 0.35:
            print("🎯 PRIMARY ISSUE: Model predictions are INVERSE")
            print("💡 SOLUTION: Invert your trading signals (BUY↔SELL)")
            print("📝 Your model learned the opposite of what you intended")
        elif overall_accuracy < 0.45:
            print("🎯 PRIMARY ISSUE: Poor directional accuracy")
            print("💡 SOLUTION: Retrain model with better features/data")
        else:
            print("🎯 Model shows directional accuracy")
            print("💡 Issue likely in risk management or execution")
            print("📝 Check stop-loss/take-profit settings")

if __name__ == "__main__":
    features, data = inspect_model_features()
    if features and data is not None:
        run_directional_accuracy_test(features, data)