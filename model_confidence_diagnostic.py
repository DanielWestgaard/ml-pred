"""
Model Confidence Diagnostic Tool

Analyzes why the model is producing 0.0000 confidence predictions
and provides recommendations for fixing the issue.
"""

import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np

def analyze_model_confidence():
    """Analyze model confidence issues from backtest results"""
    
    print("🔍 MODEL CONFIDENCE DIAGNOSTIC")
    print("=" * 50)
    
    # Find recent backtest result files
    result_pattern = "storage/models/*/backtesting/backtest_results_*.json"
    result_files = glob.glob(result_pattern)
    
    if not result_files:
        print("❌ No backtest result files found")
        return
    
    # Get most recent results
    latest_file = max(result_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"📄 Analyzing: {latest_file}")
    
    try:
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        print("\n📊 BACKTEST SUMMARY:")
        print(f"  Total Return: {results.get('total_return_pct', 'N/A')}%")
        print(f"  Win Rate: {results.get('win_rate_pct', 'N/A')}%")
        print(f"  Total Trades: {results.get('total_trades', 'N/A')}")
        print(f"  Profitable Trades: {results.get('profitable_trades', 'N/A')}")
        
        # Analyze prediction distribution if available
        if 'prediction_stats' in results:
            pred_stats = results['prediction_stats']
            print(f"\n🤖 MODEL PREDICTION ANALYSIS:")
            print(f"  Total Predictions: {pred_stats.get('total_predictions', 'N/A')}")
            print(f"  Avg Confidence: {pred_stats.get('avg_confidence', 'N/A'):.6f}")
            print(f"  Max Confidence: {pred_stats.get('max_confidence', 'N/A'):.6f}")
            print(f"  Min Confidence: {pred_stats.get('min_confidence', 'N/A'):.6f}")
            
            if 'confidence_distribution' in pred_stats:
                conf_dist = pred_stats['confidence_distribution']
                print(f"  Confidence Buckets:")
                for bucket, count in conf_dist.items():
                    print(f"    {bucket}: {count}")
        
        # Analyze signal generation
        if 'signal_stats' in results:
            sig_stats = results['signal_stats']
            print(f"\n📡 SIGNAL GENERATION ANALYSIS:")
            print(f"  Total Signals Generated: {sig_stats.get('total_signals', 'N/A')}")
            print(f"  Signals Filtered Out: {sig_stats.get('filtered_signals', 'N/A')}")
            print(f"  Signal Pass Rate: {sig_stats.get('signal_pass_rate', 'N/A')}%")
            
            # Confidence threshold analysis
            confidence_threshold = sig_stats.get('confidence_threshold', 'N/A')
            print(f"  Confidence Threshold: {confidence_threshold}")
            
            if 'rejection_reasons' in sig_stats:
                print(f"  Signal Rejection Reasons:")
                for reason, count in sig_stats['rejection_reasons'].items():
                    print(f"    {reason}: {count}")
    
    except Exception as e:
        print(f"❌ Error analyzing results file: {e}")
    
    print(f"\n🚨 CONFIDENCE PROBLEM DIAGNOSIS:")
    
    # Check if this is the flat confidence issue you mentioned
    print(f"Based on your logs showing 'confidence=0.0000', this appears to be:")
    print(f"")
    print(f"🎯 ROOT CAUSE: Model produces extremely low/zero confidence predictions")
    print(f"")
    print(f"📋 POSSIBLE CAUSES:")
    print(f"1. **Model calibration issue**: Post-SMOTE training affected confidence scores")
    print(f"2. **Feature scaling mismatch**: Training vs inference feature distributions")
    print(f"3. **Model architecture issue**: RandomForest predict_proba not working correctly")
    print(f"4. **Data preprocessing issue**: Features being processed differently at inference")
    print(f"5. **Model loading issue**: Saved model not loading correctly")
    print(f"")
    print(f"🔧 RECOMMENDED FIXES:")
    print(f"1. **Lower confidence threshold drastically**: Try 0.1 or 0.05 instead of 0.45+")
    print(f"2. **Check model predict_proba output**: Print raw model outputs during inference")
    print(f"3. **Verify feature preprocessing**: Ensure inference matches training preprocessing")
    print(f"4. **Retrain with confidence calibration**: Use CalibratedClassifierCV wrapper")
    print(f"5. **Test model on validation data**: Verify it works outside backtesting")
    print(f"")
    print(f"⚡ IMMEDIATE ACTION:")
    print(f"Run optimization with confidence_threshold range [0.01, 0.05, 0.1, 0.15, 0.2]")
    print(f"This will test if the model has useful predictions at lower thresholds.")

if __name__ == "__main__":
    analyze_model_confidence()