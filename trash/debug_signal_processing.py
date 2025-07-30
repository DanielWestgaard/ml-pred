#!/usr/bin/env python3
"""
Debug script to test the signal processing pipeline
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from utils.model.model_type_detector import PredictionProcessor, ModelType
from strategy.components.signal_processor import SignalProcessor

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
test_sample = df[feature_cols].iloc[1000:1005]

print("=== Testing Full Pipeline ===")

# Step 1: Process predictions
prediction_processor = PredictionProcessor(ModelType.MULTICLASS_CLASSIFICATION)
prediction_results = prediction_processor.process_predictions(model, test_sample)

print(f"Prediction results keys: {prediction_results.keys()}")
print(f"Predictions: {prediction_results['predictions']}")
print(f"Probabilities: {prediction_results['probabilities']}")
print(f"Confidence scores: {prediction_results['confidence_scores']}")

# Step 2: Process signals (mimic backtester.py)
signal_processor = SignalProcessor(
    confidence_threshold=0.35, 
    model_type=ModelType.MULTICLASS_CLASSIFICATION
)

predictions = prediction_results['predictions']
probabilities = prediction_results.get('positive_probabilities')  # ← This is the issue!

print(f"\n=== Signal Processor Input ===")
print(f"Predictions passed to signal processor: {predictions}")
print(f"Probabilities passed to signal processor: {probabilities}")  # This will be None!
print(f"Full probabilities available: {prediction_results['probabilities']}")

# Process signals
signals = signal_processor.process(
    predictions, 
    market_data=None,
    model=model,
    prediction_probabilities=probabilities  # ← BUG: This is None for multiclass!
)

print(f"\n=== Signal Processing Results ===")
print(f"Generated signals: {len(signals['processed_signals'])}")
print(f"Signal directions: {signals['signal_direction']}")
print(f"Signal confidences: {signals['confidence_scores']}")

# Test what happens when we pass the FULL probabilities
print(f"\n=== Testing with FULL probabilities ===")
full_probabilities = prediction_results['probabilities']
signals_fixed = signal_processor.process(
    predictions, 
    market_data=None,
    model=model,
    prediction_probabilities=full_probabilities  # ← Fix: Pass full probabilities
)

print(f"Generated signals with full probabilities: {len(signals_fixed['processed_signals'])}")
print(f"Signal directions: {signals_fixed['signal_direction']}")
print(f"Signal confidences: {signals_fixed['confidence_scores']}")