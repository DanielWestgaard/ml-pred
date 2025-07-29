#!/usr/bin/env python3
"""
Debug script to trace the exact data flow in backtester
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from utils.model.model_type_detector import PredictionProcessor, ModelType

# Load the trained model
model_path = Path("storage/models/2025-07-29_16:12:37/random_forest.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load test data
data_path = "storage/data/capital_com/GBPUSD/processed/1m/2024-01-01_2025-03-01.csv"
df = pd.read_csv(data_path)

# Get features
exclude_cols = ['date', 'target_spread_compression_v2_long_score', 
                'target_spread_compression_v2_short_score', 
                'target_spread_compression_v2_directional',
                'target_spread_compression_v2_binary']
feature_cols = [col for col in df.columns if col not in exclude_cols]
test_sample = df[feature_cols].iloc[1000:1001]  # Just 1 sample

print("=== Tracing Data Flow ===")

# Step 1: Process predictions
prediction_processor = PredictionProcessor(ModelType.MULTICLASS_CLASSIFICATION)
prediction_results = prediction_processor.process_predictions(model, test_sample)

print(f"Step 1 - Prediction Results:")
print(f"  predictions: {prediction_results['predictions']}")
print(f"  probabilities shape: {prediction_results['probabilities'].shape}")
print(f"  probabilities: {prediction_results['probabilities']}")
print(f"  positive_probabilities: {prediction_results.get('positive_probabilities')}")

# Step 2: Mimic backtester logic
model_type = 'multiclass_classification'  # This is what backtester sets

predictions = prediction_results['predictions']

# This is the EXACT logic from backtester.py
if model_type == 'multiclass_classification':
    probabilities = prediction_results.get('probabilities')
else:
    probabilities = prediction_results.get('positive_probabilities')

print(f"\nStep 2 - Backtester Logic:")
print(f"  model_type: {model_type}")
print(f"  probabilities from backtester: {probabilities}")
print(f"  probabilities shape: {probabilities.shape if probabilities is not None else 'None'}")

# Step 3: What signal processor receives
if probabilities is not None:
    prob_for_prediction_0 = probabilities[0]
    print(f"\nStep 3 - Signal Processor Input:")
    print(f"  probabilities[0]: {prob_for_prediction_0}")
    print(f"  probabilities[0] type: {type(prob_for_prediction_0)}")
    print(f"  len(probabilities[0]): {len(prob_for_prediction_0)}")
    
    # Test the signal processor logic
    if hasattr(prob_for_prediction_0, '__len__') and len(prob_for_prediction_0) == 3:
        print(f"  ✅ Signal processor will correctly use probabilities")
        prob_hold, prob_buy, prob_sell = prob_for_prediction_0
        predicted_class = int(predictions[0])
        
        if predicted_class == 0:  # HOLD
            confidence = prob_hold
            print(f"  Class 0 (HOLD): confidence = {confidence:.4f}")
        elif predicted_class == 1:  # BUY
            confidence = prob_buy
            print(f"  Class 1 (BUY): confidence = {confidence:.4f}")
        elif predicted_class == 2:  # SELL
            confidence = prob_sell
            print(f"  Class 2 (SELL): confidence = {confidence:.4f}")
    else:
        print(f"  ❌ Signal processor will use fallback logic")
else:
    print(f"\nStep 3 - Signal Processor Input:")
    print(f"  ❌ probabilities is None - will use fallback hardcoded values")