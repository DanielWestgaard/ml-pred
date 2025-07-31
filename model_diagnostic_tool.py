#!/usr/bin/env python3
"""
Model Diagnostic Tool

Comprehensive diagnostic tool to analyze model confidence distributions, 
feature preprocessing consistency, and identify root causes of low confidence predictions.

Usage:
    python model_diagnostic_tool.py <model_path> <data_path> [--config <config_path>]
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logging_utils import get_logger
from utils import data_utils
from utils.model import model_utils
import utils.hypothesis.config_loader as config_loader

logger = get_logger()

class ModelDiagnostic:
    """Comprehensive model diagnostic tool for analyzing prediction confidence issues."""
    
    def __init__(self, model_path: str, data_path: str, config_path: Optional[str] = None):
        self.model_path = model_path
        self.data_path = data_path
        self.config_path = config_path
        self.model = None
        self.data = None
        self.X = None
        self.y = None
        self.predictions = None
        self.confidence_scores = None
        
    def load_model_and_data(self):
        """Load the trained model and corresponding data."""
        logger.info("🔍 Loading model and data for diagnostic analysis...")
        
        # Load model
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✅ Successfully loaded model from: {self.model_path}")
            logger.info(f"Model type: {type(self.model).__name__}")
            
            # Check if model has expected methods
            if not hasattr(self.model, 'predict_proba'):
                logger.error("❌ Model does not have predict_proba method")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
        
        # Load data
        try:
            self.data, _ = data_utils.check_and_return_df(self.data_path)
            logger.info(f"✅ Successfully loaded data from: {self.data_path}")
            logger.info(f"Data shape: {self.data.shape}")
            
            # Set up datetime index if needed
            if self.data.index.name != 'date' and 'date' in self.data.columns:
                self.data.index = pd.to_datetime(self.data['date'])
                self.data = self.data.drop('date', axis=1)
                
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return False
            
        return True
    
    def prepare_features_and_targets(self):
        """Prepare features and targets using the same logic as model training."""
        logger.info("🔧 Preparing features and targets...")
        
        try:
            # Load model metadata to get exact feature list used during training
            metadata_path = os.path.join(os.path.dirname(self.model_path), 'model_metadata.yml')
            expected_features = None
            metadata = None
            
            if os.path.exists(metadata_path):
                try:
                    import yaml
                    with open(metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                    expected_features = metadata.get('dataset', {}).get('features', [])
                    logger.info(f"Loading expected features from metadata: {expected_features}")
                except Exception as e:
                    logger.warning(f"Could not load metadata: {e}")
            
            # Get target column from model metadata
            target_col = None
            if metadata:
                target_col = metadata.get('dataset', {}).get('target_column')
                logger.info(f"Using target column from metadata: {target_col}")
            
            # Fallback to configuration
            if not target_col and self.config_path:
                configurations = config_loader.load_hypothesis_config(self.config_path).handle_hypothesis_source()
                target_col = configurations.get("model_training_target_column", "target_spread_compression_patterns_v3_directional")
            
            # Final fallback
            if not target_col:
                target_col = "target_spread_compression_patterns_v3_directional"
            
            # Handle categories
            self.data = model_utils.handle_categories(df=self.data)
            
            # Use exact features from training if available
            if expected_features:
                # Check if all expected features are available
                missing_features = [f for f in expected_features if f not in self.data.columns]
                if missing_features:
                    logger.error(f"❌ Missing expected features: {missing_features}")
                    logger.info(f"Available columns: {list(self.data.columns)}")
                    return False
                
                # Use only the exact features used during training
                self.X = self.data[expected_features].copy()
                logger.info(f"Using exact training features: {expected_features}")
            else:
                # Fallback to previous logic if metadata not available
                logger.warning("No feature metadata found, using fallback feature selection")
                target_columns = [col for col in self.data.columns if col.startswith('target_')]
                future_columns = [col for col in self.data.columns if col.startswith('future_')]
                exclude_columns = target_columns + future_columns
                self.X = self.data.drop(exclude_columns, axis=1, errors='ignore')
            
            # Prepare target
            target_columns = [col for col in self.data.columns if col.startswith('target_')]
            if target_col in self.data.columns:
                self.y = self.data[target_col]
                logger.info(f"Using target column: {target_col}")
            else:
                logger.warning(f"Target column '{target_col}' not found. Available targets: {target_columns}")
                if target_columns:
                    self.y = self.data[target_columns[0]]
                    logger.info(f"Using first available target: {target_columns[0]}")
                else:
                    logger.error("No target columns found!")
                    return False
            
            logger.info(f"Features shape: {self.X.shape}")
            logger.info(f"Target shape: {self.y.shape}")
            logger.info(f"Feature columns: {list(self.X.columns)}")
            logger.info(f"Target distribution: {self.y.value_counts().to_dict()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare features and targets: {e}")
            return False
    
    def analyze_model_predictions(self):
        """Analyze model predictions and confidence distributions."""
        logger.info("📊 Analyzing model predictions and confidence distributions...")
        
        try:
            # Get predictions and confidence scores
            self.predictions = self.model.predict(self.X)
            self.confidence_scores = self.model.predict_proba(self.X)
            
            logger.info(f"Predictions shape: {self.predictions.shape}")
            logger.info(f"Confidence scores shape: {self.confidence_scores.shape}")
            
            # Analyze predictions
            pred_counts = pd.Series(self.predictions).value_counts()
            logger.info(f"Prediction distribution: {pred_counts.to_dict()}")
            
            # Analyze confidence scores
            logger.info("\n🎯 CONFIDENCE SCORE ANALYSIS:")
            
            for class_idx in range(self.confidence_scores.shape[1]):
                class_confidences = self.confidence_scores[:, class_idx]
                logger.info(f"\nClass {class_idx} confidence scores:")
                logger.info(f"  Min: {class_confidences.min():.6f}")
                logger.info(f"  Max: {class_confidences.max():.6f}")
                logger.info(f"  Mean: {class_confidences.mean():.6f}")
                logger.info(f"  Std: {class_confidences.std():.6f}")
                logger.info(f"  Unique values: {len(np.unique(class_confidences))}")
                
                # Check for constant values (key diagnostic indicator)
                if len(np.unique(class_confidences)) <= 3:
                    logger.warning(f"⚠️ Class {class_idx} has very few unique confidence values! This indicates a broken model.")
                    logger.info(f"  Unique values: {np.unique(class_confidences)}")
            
            # Analyze maximum confidence per prediction
            max_confidences = np.max(self.confidence_scores, axis=1)
            logger.info(f"\n🔍 MAXIMUM CONFIDENCE ANALYSIS:")
            logger.info(f"  Min max confidence: {max_confidences.min():.6f}")
            logger.info(f"  Max max confidence: {max_confidences.max():.6f}")
            logger.info(f"  Mean max confidence: {max_confidences.mean():.6f}")
            logger.info(f"  Unique max confidence values: {len(np.unique(max_confidences))}")
            
            # Check thresholds
            thresholds = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.485, 0.5]
            logger.info(f"\n📈 THRESHOLD ANALYSIS:")
            for threshold in thresholds:
                above_threshold = np.sum(max_confidences > threshold)
                percentage = (above_threshold / len(max_confidences)) * 100
                logger.info(f"  Above {threshold}: {above_threshold} samples ({percentage:.2f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze predictions: {e}")
            return False
    
    def diagnose_smote_impact(self):
        """Diagnose potential SMOTE-related issues."""
        logger.info("\n🔬 DIAGNOSING SMOTE IMPACT...")
        
        # Check if model was trained with SMOTE
        if hasattr(self.model, 'feature_importances_'):
            feature_importances = self.model.feature_importances_
            
            # Check for unusual feature importance patterns
            zero_importance = np.sum(feature_importances == 0)
            very_low_importance = np.sum(feature_importances < 0.001)
            
            logger.info(f"Features with zero importance: {zero_importance}")
            logger.info(f"Features with very low importance (<0.001): {very_low_importance}")
            
            if zero_importance > len(feature_importances) * 0.5:
                logger.warning("⚠️ More than 50% of features have zero importance - possible SMOTE overfitting")
            
            # Show top and bottom features
            feature_df = pd.DataFrame({
                'feature': self.X.columns,
                'importance': feature_importances
            }).sort_values('importance', ascending=False)
            
            logger.info(f"\nTop 5 most important features:")
            for _, row in feature_df.head(5).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.6f}")
                
            logger.info(f"\nBottom 5 least important features:")
            for _, row in feature_df.tail(5).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.6f}")
    
    def check_feature_preprocessing_consistency(self):
        """Check for feature preprocessing consistency issues."""
        logger.info("\n🔍 CHECKING FEATURE PREPROCESSING CONSISTENCY...")
        
        try:
            # Check for NaN values
            nan_counts = self.X.isnull().sum()
            total_nans = nan_counts.sum()
            
            if total_nans > 0:
                logger.warning(f"⚠️ Found {total_nans} NaN values in features!")
                nan_features = nan_counts[nan_counts > 0]
                for feature, count in nan_features.items():
                    logger.warning(f"  {feature}: {count} NaN values")
            else:
                logger.info("✅ No NaN values found in features")
            
            # Check for infinite values
            inf_counts = np.isinf(self.X.select_dtypes(include=[np.number])).sum()
            total_infs = inf_counts.sum()
            
            if total_infs > 0:
                logger.warning(f"⚠️ Found {total_infs} infinite values in features!")
                inf_features = inf_counts[inf_counts > 0]
                for feature, count in inf_features.items():
                    logger.warning(f"  {feature}: {count} infinite values")
            else:
                logger.info("✅ No infinite values found in features")
            
            # Check data types
            logger.info("\n📋 FEATURE DATA TYPES:")
            dtype_counts = self.X.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                logger.info(f"  {dtype}: {count} features")
            
            # Check for constant features
            constant_features = []
            for col in self.X.columns:
                if self.X[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                logger.warning(f"⚠️ Found {len(constant_features)} constant features:")
                for feature in constant_features[:5]:  # Show first 5
                    logger.warning(f"  {feature}")
            else:
                logger.info("✅ No constant features found")
            
            # Check value ranges
            numeric_features = self.X.select_dtypes(include=[np.number])
            logger.info(f"\n📊 FEATURE VALUE RANGES (sample of 5):")
            for col in numeric_features.columns[:5]:
                logger.info(f"  {col}: [{numeric_features[col].min():.6f}, {numeric_features[col].max():.6f}]")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to check feature preprocessing: {e}")
            return False
    
    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report with recommendations."""
        logger.info("\n" + "="*80)
        logger.info("🎯 COMPREHENSIVE DIAGNOSTIC REPORT")
        logger.info("="*80)
        
        # Root cause analysis
        max_confidences = np.max(self.confidence_scores, axis=1)
        unique_max_confidences = len(np.unique(max_confidences))
        mean_max_confidence = max_confidences.mean()
        above_001 = np.sum(max_confidences > 0.001)
        above_01 = np.sum(max_confidences > 0.01)
        above_485 = np.sum(max_confidences > 0.485)
        
        logger.info(f"\n🔍 ROOT CAUSE ANALYSIS:")
        
        if unique_max_confidences <= 5:
            logger.error(f"❌ CRITICAL: Model outputs only {unique_max_confidences} unique confidence values")
            logger.error("   This indicates severe overfitting or corrupted training")
            
        if mean_max_confidence < 0.1:
            logger.error(f"❌ CRITICAL: Mean confidence is extremely low ({mean_max_confidence:.6f})")
            logger.error("   Model has no confidence in its predictions")
            
        if above_485 < 10:
            logger.error(f"❌ CRITICAL: Only {above_485} predictions above 0.485 confidence threshold")
            logger.error("   This explains why backtesting generates almost no trades")
        
        # Recommendations
        logger.info(f"\n💡 RECOMMENDATIONS:")
        
        if unique_max_confidences <= 5 and mean_max_confidence < 0.1:
            logger.info("1. 🔄 IMMEDIATE: Retrain model from scratch")
            logger.info("   - The current model is fundamentally broken")
            logger.info("   - SMOTE may have caused severe overfitting")
            
        if above_485 < 100:
            logger.info("2. ⚙️ URGENT: Adjust confidence thresholds")
            logger.info(f"   - Current 0.485 threshold is too high")
            logger.info(f"   - Consider threshold around 0.01-0.05 range")
            logger.info(f"   - {above_01} predictions above 0.01 threshold")
            
        logger.info("3. 🔬 INVESTIGATE: Check SMOTE implementation")
        logger.info("   - SMOTE may be creating unrealistic synthetic samples")
        logger.info("   - Consider using different resampling strategies")
        
        logger.info("4. 📊 VALIDATE: Feature engineering pipeline")
        logger.info("   - Ensure features are properly normalized/scaled")
        logger.info("   - Check for data leakage in feature creation")
        
        # Model health summary
        logger.info(f"\n🏥 MODEL HEALTH SUMMARY:")
        
        health_score = 0
        max_score = 4
        
        if unique_max_confidences > 10:
            health_score += 1
            logger.info("✅ Confidence diversity: GOOD")
        else:
            logger.info("❌ Confidence diversity: POOR")
            
        if mean_max_confidence > 0.3:
            health_score += 1
            logger.info("✅ Average confidence: GOOD")
        else:
            logger.info("❌ Average confidence: POOR")
            
        if above_485 > 1000:
            health_score += 1
            logger.info("✅ High confidence predictions: GOOD")
        else:
            logger.info("❌ High confidence predictions: POOR")
            
        if not hasattr(self, 'critical_issues'):
            health_score += 1
            logger.info("✅ Data quality: GOOD")
        else:
            logger.info("❌ Data quality: POOR")
            
        health_percentage = (health_score / max_score) * 100
        logger.info(f"\n🎯 OVERALL MODEL HEALTH: {health_percentage:.0f}% ({health_score}/{max_score})")
        
        if health_percentage < 50:
            logger.error("🚨 MODEL REQUIRES IMMEDIATE RETRAINING")
        elif health_percentage < 75:
            logger.warning("⚠️ MODEL NEEDS SIGNIFICANT IMPROVEMENTS")
        else:
            logger.info("✅ MODEL IS HEALTHY")
    
    def run_full_diagnostic(self):
        """Run complete diagnostic analysis."""
        logger.info("🚀 Starting comprehensive model diagnostic analysis...")
        
        if not self.load_model_and_data():
            return False
            
        if not self.prepare_features_and_targets():
            return False
            
        if not self.analyze_model_predictions():
            return False
            
        self.diagnose_smote_impact()
        self.check_feature_preprocessing_consistency()
        self.generate_diagnostic_report()
        
        logger.info("\n✅ Diagnostic analysis completed!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Model Diagnostic Tool')
    parser.add_argument('model_path', help='Path to the trained model (.pkl file)')
    parser.add_argument('data_path', help='Path to the data file')
    parser.add_argument('--config', help='Path to hypothesis config file (optional)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return False
        
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return False
    
    # Run diagnostic
    diagnostic = ModelDiagnostic(args.model_path, args.data_path, args.config)
    return diagnostic.run_full_diagnostic()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)