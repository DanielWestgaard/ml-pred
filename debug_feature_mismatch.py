#!/usr/bin/env python3
"""
Feature Mismatch Diagnostic
==========================

Systematically compare features between model training and backtesting to identify
why the model returns 0.0000 confidence during backtesting.

HYPOTHESIS: Feature preprocessing or data format differences between training and backtesting.
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

def load_model_and_features():
    """Load the trained model and extract feature information."""
    
    model_path = Path("storage/models/2025-07-31_19:16:38")
    
    # Load model metadata
    with open(model_path / "model_metadata.yml", 'r') as f:
        metadata = yaml.safe_load(f)
    
    # Load model
    with open(model_path / "random_forest.pkl", 'rb') as f:
        model = pickle.load(f)
    
    training_features = metadata['dataset']['features']
    logger.info(f"✅ Model loaded: {len(training_features)} features")
    logger.info(f"Training features: {training_features}")
    
    return model, metadata, training_features

def analyze_training_data():
    """Analyze the training dataset features."""
    
    training_data_path = "storage/data/capital_com/GBPUSD/processed/1m/2025-03-02_2025-07-01.csv"
    
    logger.info(f"📊 TRAINING DATA ANALYSIS:")
    logger.info(f"Loading: {training_data_path}")
    
    df_train = pd.read_csv(training_data_path)
    logger.info(f"Shape: {df_train.shape}")
    logger.info(f"Columns: {list(df_train.columns)}")
    
    # Check training features
    model, metadata, training_features = load_model_and_features()
    
    logger.info(f"\n🔍 FEATURE AVAILABILITY IN TRAINING DATA:")
    for feature in training_features:
        if feature in df_train.columns:
            sample_values = df_train[feature].dropna().head(5).values
            logger.info(f"  ✅ {feature}: Available, sample values: {sample_values}")
        else:
            logger.info(f"  ❌ {feature}: MISSING from training data!")
    
    return df_train, training_features

def analyze_backtesting_data():
    """Analyze the backtesting dataset features."""
    
    # Same data source as training, but let's see what backtesting gets
    backtesting_data_path = "storage/data/capital_com/GBPUSD/processed/1m/2025-03-02_2025-07-01.csv"
    
    logger.info(f"\n📊 BACKTESTING DATA ANALYSIS:")
    logger.info(f"Loading: {backtesting_data_path}")
    
    df_backtest = pd.read_csv(backtesting_data_path)
    
    # Filter to backtesting period (2025-03-10 to 2025-03-20)
    df_backtest['date'] = pd.to_datetime(df_backtest['date'])
    df_backtest_filtered = df_backtest[
        (df_backtest['date'] >= '2025-03-10') & 
        (df_backtest['date'] <= '2025-03-20')
    ].copy()
    
    logger.info(f"Backtesting period shape: {df_backtest_filtered.shape}")
    logger.info(f"Date range: {df_backtest_filtered['date'].min()} to {df_backtest_filtered['date'].max()}")
    
    return df_backtest_filtered

def test_model_predictions():
    """Test model predictions on actual backtesting data to identify the issue."""
    
    logger.info(f"\n🤖 MODEL PREDICTION TEST:")
    
    # Load model and backtesting data
    model, metadata, training_features = load_model_and_features()
    df_backtest = analyze_backtesting_data()
    
    # Check feature availability in backtesting data
    logger.info(f"\n🔍 FEATURE AVAILABILITY IN BACKTESTING DATA:")
    missing_features = []
    
    for feature in training_features:
        if feature in df_backtest.columns:
            sample_values = df_backtest[feature].dropna().head(5).values
            logger.info(f"  ✅ {feature}: Available, sample values: {sample_values}")
        else:
            logger.info(f"  ❌ {feature}: MISSING from backtesting data!")
            missing_features.append(feature)
    
    if missing_features:
        logger.info(f"\n❌ CRITICAL: Missing features in backtesting data: {missing_features}")
        return
    
    # Test predictions on first 100 rows
    logger.info(f"\n🎯 TESTING MODEL PREDICTIONS:")
    test_data = df_backtest[training_features].head(100)
    
    # Check for data quality issues
    logger.info(f"Test data shape: {test_data.shape}")
    logger.info(f"Null values: {test_data.isnull().sum().sum()}")
    logger.info(f"Infinite values: {np.isinf(test_data.values).sum()}")
    
    # Clean data if needed
    if test_data.isnull().sum().sum() > 0:
        logger.info("⚠️  Filling null values with 0")
        test_data = test_data.fillna(0)
    
    if np.isinf(test_data.values).sum() > 0:
        logger.info("⚠️  Replacing infinite values with 0")
        test_data = test_data.replace([np.inf, -np.inf], 0)
    
    try:
        # Get predictions
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)
        
        # Calculate confidence (max probability)
        max_probas = np.max(probabilities, axis=1)
        
        logger.info(f"\n📊 PREDICTION RESULTS:")
        logger.info(f"Prediction shape: {predictions.shape}")
        logger.info(f"Probability shape: {probabilities.shape}")
        logger.info(f"Classes: {model.classes_}")
        
        # Analyze predictions
        pred_dist = pd.Series(predictions).value_counts().sort_index()
        logger.info(f"Prediction distribution: {pred_dist.to_dict()}")
        
        logger.info(f"\n📊 CONFIDENCE ANALYSIS:")
        logger.info(f"Min confidence: {max_probas.min():.6f}")
        logger.info(f"Max confidence: {max_probas.max():.6f}")
        logger.info(f"Mean confidence: {max_probas.mean():.6f}")
        logger.info(f"Median confidence: {np.median(max_probas):.6f}")
        
        # Check critical confidence thresholds
        high_conf_001 = (max_probas > 0.01).sum()
        high_conf_01 = (max_probas > 0.1).sum()
        high_conf_03 = (max_probas > 0.3).sum()
        
        logger.info(f"\n🚨 CONFIDENCE THRESHOLD ANALYSIS:")
        logger.info(f"Predictions with confidence > 0.01: {high_conf_001}/{len(max_probas)} ({high_conf_001/len(max_probas)*100:.1f}%)")
        logger.info(f"Predictions with confidence > 0.1:  {high_conf_01}/{len(max_probas)} ({high_conf_01/len(max_probas)*100:.1f}%)")
        logger.info(f"Predictions with confidence > 0.3:  {high_conf_03}/{len(max_probas)} ({high_conf_03/len(max_probas)*100:.1f}%)")
        
        if high_conf_001 == 0:
            logger.info("❌ CRITICAL: ALL predictions have confidence ≤ 0.01!")
            logger.info("   This explains why backtesting rejects all signals.")
        elif high_conf_001 < len(max_probas) * 0.1:
            logger.info("⚠️  WARNING: Very few predictions have meaningful confidence")
        else:
            logger.info("✅ Model generating reasonable confidence values")
        
        # Sample detailed predictions
        logger.info(f"\n📝 SAMPLE PREDICTIONS (first 10):")
        for i in range(min(10, len(predictions))):
            logger.info(f"  Row {i}: Pred={predictions[i]}, Conf={max_probas[i]:.6f}, Proba={probabilities[i]}")
        
        # Compare with training data statistics
        logger.info(f"\n🔄 COMPARISON WITH MODEL DIAGNOSTIC:")
        logger.info("Earlier model diagnostic showed confidence range 0.345-0.999")
        logger.info("If this test shows much lower confidence, there's a data issue")
        
    except Exception as e:
        logger.info(f"❌ ERROR during model prediction: {e}")
        import traceback
        traceback.print_exc()

def run_comprehensive_feature_diagnostic():
    """Run complete feature mismatch diagnostic."""
    
    logger.info("🚀 STARTING FEATURE MISMATCH DIAGNOSTIC")
    logger.info("="*80)
    logger.info("Systematically comparing training vs backtesting features")
    logger.info("to identify why model returns 0.0000 confidence")
    logger.info("="*80)
    
    # Step 1: Analyze training data
    df_train, training_features = analyze_training_data()
    
    # Step 2: Analyze backtesting data  
    df_backtest = analyze_backtesting_data()
    
    # Step 3: Test model predictions
    test_model_predictions()
    
    logger.info("\n" + "="*80)
    logger.info("🎯 DIAGNOSTIC COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    run_comprehensive_feature_diagnostic()