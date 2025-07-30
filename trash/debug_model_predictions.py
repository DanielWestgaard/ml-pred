#!/usr/bin/env python3
"""
Debug script to test model predictions directly
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Load the trained model
model_path = Path("storage/models/2025-07-29_16:12:37/random_forest.pkl")
print(f"Loading model from: {model_path}")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"Model type: {type(model)}")
print(f"Model has predict_proba: {hasattr(model, 'predict_proba')}")

# Load some test data
data_path = "storage/data/capital_com/GBPUSD/processed/1m/2024-01-01_2025-03-01.csv"
print(f"\nLoading test data from: {data_path}")

df = pd.read_csv(data_path)
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Get features that don't contain target or future data
exclude_cols = ['date', 'target_spread_compression_v2_long_score', 
                'target_spread_compression_v2_short_score', 
                'target_spread_compression_v2_directional',
                'target_spread_compression_v2_binary']

feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"\nUsing {len(feature_cols)} features")
print(f"Features: {feature_cols}")

# Get a small sample for testing
test_sample = df[feature_cols].iloc[1000:1005]  # 5 samples
print(f"\nTest sample shape: {test_sample.shape}")

# Make predictions
print("\n=== Testing Model Predictions ===")
predictions = model.predict(test_sample)
print(f"Predictions: {predictions}")
print(f"Unique predictions: {np.unique(predictions)}")

# Get probabilities
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(test_sample)
    print(f"\nProbabilities shape: {probabilities.shape}")
    print(f"Probabilities:\n{probabilities}")
    
    # Check if these are the problematic fixed values
    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        max_prob = np.max(probs)
        pred_class = np.argmax(probs)
        print(f"Sample {i}: Prediction={pred}, Max_prob={max_prob:.4f}, Predicted_class={pred_class}")
        
        # Check for suspicious fixed values
        if max_prob == 0.8000 or max_prob == 0.0500:
            print(f"  ⚠️  SUSPICIOUS: Fixed probability detected!")

print("\n=== Testing Direct Signal Processing ===")
# Test the problematic pipeline directly
from utils.model.model_type_detector import PredictionProcessor, ModelType

processor = PredictionProcessor(ModelType.MULTICLASS_CLASSIFICATION)
result = processor.process_predictions(model, test_sample)

print(f"Processor predictions: {result['predictions']}")
print(f"Processor probabilities shape: {result['probabilities'].shape if result['probabilities'] is not None else 'None'}")
print(f"Processor confidence scores: {result['confidence_scores']}")