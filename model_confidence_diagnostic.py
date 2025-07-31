#!/usr/bin/env python3
"""
Model Confidence Calibration Diagnostic
=======================================

Investigates why the trained model is outputting 0.0000 confidence for all predictions.
This is critical for identifying confidence calibration and class imbalance issues.
"""

import pandas as pd
import numpy as np
import pickle
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.logging_utils import get_logger

logger = get_logger()

def load_model_and_metadata(model_path):
    """Load the trained model and its metadata"""
    
    model_path = Path(model_path)
    
    # Load model metadata
    metadata_path = model_path / "model_metadata.yml"
    if not metadata_path.exists():
        print(f"❌ Model metadata not found: {metadata_path}")
        return None, None
    
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    # Load the actual model (try both naming conventions)
    model_file = model_path / "model.pkl"
    if not model_file.exists():
        model_file = model_path / "random_forest.pkl"
    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        return None, metadata
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Loaded model: {type(model).__name__}")
    print(f"✅ Model created: {metadata['model_info']['created_at']}")
    
    return model, metadata

def analyze_training_data_distribution(metadata):
    """Analyze the target distribution in training data"""
    
    print("\n" + "="*60)
    print("🎯 TRAINING DATA TARGET DISTRIBUTION ANALYSIS")
    print("="*60)
    
    dataset_path = metadata['dataset']['dataset_path']
    target_column = metadata['dataset']['target_column']
    
    print(f"Dataset: {dataset_path}")
    print(f"Target: {target_column}")
    
    if not Path(dataset_path).exists():
        print(f"❌ Training dataset not found: {dataset_path}")
        return None
    
    # Load training data
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    
    if target_column not in df.columns:
        print(f"❌ Target column not found: {target_column}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Analyze target distribution
    target_dist = df[target_column].value_counts().sort_index()
    total_samples = len(df)
    
    print(f"\n📊 TARGET DISTRIBUTION:")
    print(f"Total samples: {total_samples:,}")
    
    for class_val, count in target_dist.items():
        percentage = count / total_samples * 100
        class_name = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(class_val, f'Class_{class_val}')
        print(f"  {class_name} (Class {class_val}): {count:,} samples ({percentage:.2f}%)")
    
    # Check for extreme imbalance
    if len(target_dist) > 1:
        min_class_pct = min(target_dist) / total_samples * 100
        max_class_pct = max(target_dist) / total_samples * 100
        imbalance_ratio = max_class_pct / min_class_pct
        
        print(f"\n⚖️  CLASS BALANCE ANALYSIS:")
        print(f"Most common class: {max_class_pct:.2f}%")
        print(f"Least common class: {min_class_pct:.2f}%") 
        print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 50:
            print(f"❌ SEVERE IMBALANCE: {imbalance_ratio:.1f}:1 ratio may cause confidence issues!")
        elif imbalance_ratio > 10:
            print(f"⚠️  MODERATE IMBALANCE: {imbalance_ratio:.1f}:1 ratio")
        else:
            print(f"✅ REASONABLE BALANCE: {imbalance_ratio:.1f}:1 ratio")
    
    return df

def test_model_predictions(model, metadata, df):
    """Test model predictions and confidence calibration"""
    
    print("\n" + "="*60)
    print("🤖 MODEL PREDICTION CONFIDENCE ANALYSIS")
    print("="*60)
    
    if model is None or df is None:
        print("❌ Cannot test - model or data not available")
        return
    
    # Get feature columns from metadata
    feature_cols = metadata['dataset']['features']
    target_col = metadata['dataset']['target_column']
    
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")
    
    # Check if all features are available
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        return
    
    # Take a sample for testing (last 1000 rows)
    df_test = df.tail(1000).copy()
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]
    
    print(f"\nTesting on {len(df_test)} samples...")
    
    # Check for any data issues
    print(f"\n🔍 DATA QUALITY CHECK:")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_test null values: {X_test.isnull().sum().sum()}")
    print(f"X_test infinite values: {np.isinf(X_test.values).sum()}")
    
    if X_test.isnull().sum().sum() > 0:
        print("⚠️  NULL VALUES DETECTED - filling with 0")
        X_test = X_test.fillna(0)
    
    if np.isinf(X_test.values).sum() > 0:
        print("⚠️  INFINITE VALUES DETECTED - replacing")
        X_test = X_test.replace([np.inf, -np.inf], 0)
    
    try:
        # Test basic predictions
        print(f"\n🎯 PREDICTION ANALYSIS:")
        
        # Get class predictions
        y_pred = model.predict(X_test)
        print(f"Predictions shape: {y_pred.shape}")
        
        pred_dist = pd.Series(y_pred).value_counts().sort_index()
        print(f"Prediction distribution: {pred_dist.to_dict()}")
        
        # Get prediction probabilities (this is key for confidence!)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            print(f"Probabilities shape: {y_proba.shape}")
            print(f"Probability classes: {model.classes_}")
            
            # Analyze confidence distribution
            max_probas = np.max(y_proba, axis=1)  # Confidence = max probability
            
            print(f"\n📊 CONFIDENCE DISTRIBUTION:")
            print(f"Min confidence: {max_probas.min():.6f}")
            print(f"Max confidence: {max_probas.max():.6f}")
            print(f"Mean confidence: {max_probas.mean():.6f}")
            print(f"Median confidence: {np.median(max_probas):.6f}")
            
            # Check confidence percentiles
            for percentile in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                conf_val = np.percentile(max_probas, percentile)
                print(f"{percentile:2d}th percentile: {conf_val:.6f}")
            
            # Critical check: How many predictions have confidence > 0.01?
            high_conf_count = (max_probas > 0.01).sum()
            high_conf_pct = high_conf_count / len(max_probas) * 100
            
            print(f"\n🚨 CRITICAL CONFIDENCE ANALYSIS:")
            print(f"Predictions with confidence > 0.01: {high_conf_count}/{len(max_probas)} ({high_conf_pct:.1f}%)")
            
            if high_conf_pct < 1:
                print("❌ CRITICAL ISSUE: Almost no predictions have meaningful confidence!")
                print("   This explains why all signals are rejected during backtesting.")
            elif high_conf_pct < 10:
                print("⚠️  WARNING: Very few predictions have meaningful confidence")
            else:
                print("✅ Reasonable number of confident predictions")
            
            # Analyze by prediction class
            print(f"\n📊 CONFIDENCE BY PREDICTION CLASS:")
            for class_val in model.classes_:
                class_mask = (y_pred == class_val)
                if class_mask.sum() > 0:
                    class_conf = max_probas[class_mask]
                    class_name = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(class_val, f'Class_{class_val}')
                    print(f"  {class_name}: {class_mask.sum()} predictions, "
                          f"avg confidence: {class_conf.mean():.6f}, "
                          f"max confidence: {class_conf.max():.6f}")
            
            # Check if model is only predicting one class
            unique_preds = len(pred_dist)
            if unique_preds == 1:
                print(f"\n❌ CRITICAL: Model only predicting class {pred_dist.index[0]}!")
                print("   This indicates severe class imbalance learning issues.")
            
        else:
            print("❌ Model doesn't support predict_proba - cannot analyze confidence")
        
    except Exception as e:
        print(f"❌ Error during prediction testing: {e}")

def analyze_model_training_process(metadata):
    """Analyze the model training process for issues"""
    
    print("\n" + "="*60)
    print("🏋️  MODEL TRAINING PROCESS ANALYSIS") 
    print("="*60)
    
    # Check model parameters
    model_params = metadata.get('model_parameters', {})
    print(f"📋 MODEL PARAMETERS:")
    
    key_params = ['class_weight', 'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
    for param in key_params:
        if param in model_params:
            print(f"  {param}: {model_params[param]}")
    
    # Check if class_weight is balanced
    class_weight = model_params.get('class_weight')
    if class_weight == 'balanced':
        print("✅ Using balanced class weights")
    elif class_weight is None:
        print("⚠️  WARNING: No class weighting - may cause imbalance issues")
    else:
        print(f"🔧 Custom class weights: {class_weight}")
    
    # Check training data size
    train_samples = metadata.get('training', {}).get('train_samples', 0)
    test_samples = metadata.get('training', {}).get('test_samples', 0)
    
    print(f"\n📊 TRAINING DATA SIZE:")
    print(f"  Training samples: {train_samples:,}")
    print(f"  Test samples: {test_samples:,}")
    
    if train_samples < 10000:
        print("⚠️  WARNING: Small training set may cause poor calibration")

def run_comprehensive_confidence_analysis():
    """Run complete confidence calibration analysis"""
    
    print("🚀 STARTING MODEL CONFIDENCE CALIBRATION ANALYSIS")
    print("="*80)
    print("This analysis will identify why the model has 0.0000 confidence")
    print("for all predictions during backtesting.")
    print("="*80)
    
    # Use the latest trained model with improved class balance
    model_path = "storage/models/2025-07-31_19:16:38"
    
    if not Path(model_path).exists():
        print(f"❌ Model path not found: {model_path}")
        return
    
    # Load model and metadata
    model, metadata = load_model_and_metadata(model_path)
    
    if metadata is None:
        print("❌ Cannot proceed without metadata")
        return
    
    # Analyze training data distribution
    df = analyze_training_data_distribution(metadata)
    
    # Analyze model training process
    analyze_model_training_process(metadata)
    
    # Test model predictions and confidence
    test_model_predictions(model, metadata, df)
    
    print("\n" + "="*80)
    print("🎯 CONFIDENCE ANALYSIS COMPLETE")
    print("="*80)
    print("Key findings:")
    print("1. Check if target distribution is severely imbalanced")
    print("2. Check if model confidence is consistently near zero")
    print("3. Check if model is only predicting one class")
    print("4. Review class weighting and training parameters")
    print("="*80)

if __name__ == "__main__":
    run_comprehensive_confidence_analysis()