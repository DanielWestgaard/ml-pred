#!/usr/bin/env python3
"""
Systematic Model Diagnostic Framework
Identifies exactly where the model is failing and why 0% win rate occurs
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from utils.logging_utils import get_logger
from data.features.enhanced_generation import ModelAwareFeatureGenerator
from data.features.target_hypothesis import spread_compression_patterns_v3

logger = get_logger()

class ModelDiagnosticFramework:
    """Comprehensive framework to diagnose model prediction issues"""
    
    def __init__(self, model_folder: str, data_file: str):
        """
        Initialize diagnostic framework
        
        Args:
            model_folder: Path to trained model
            data_file: Path to backtesting data file
        """
        self.model_folder = Path(model_folder)
        self.data_file = Path(data_file)
        self.model = None
        self.data = None
        self.features = None
        self.targets = None
        self.predictions = None
        self.probabilities = None
        
        logger.info(f"🔬 Model Diagnostic Framework initialized")
        logger.info(f"   Model: {model_folder}")
        logger.info(f"   Data: {data_file}")
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run complete diagnostic suite"""
        
        print("🔬 SYSTEMATIC MODEL DIAGNOSTIC")
        print("=" * 60)
        
        results = {}
        
        # Step 1: Load and validate data
        print("\n📊 STEP 1: Data Loading & Validation")
        print("-" * 40)
        results['data_loading'] = self._test_data_loading()
        
        # Step 2: Model prediction testing
        print("\n🤖 STEP 2: Model Prediction Analysis")
        print("-" * 40)
        results['prediction_analysis'] = self._test_model_predictions()
        
        # Step 3: Directional accuracy testing
        print("\n🎯 STEP 3: Directional Accuracy Analysis")
        print("-" * 40)
        results['directional_accuracy'] = self._test_directional_accuracy()
        
        # Step 4: Feature-target relationship testing
        print("\n🔗 STEP 4: Feature-Target Relationship Analysis")
        print("-" * 40)
        results['feature_target_analysis'] = self._test_feature_target_relationships()
        
        # Step 5: Temporal consistency testing
        print("\n⏰ STEP 5: Temporal Consistency Analysis")
        print("-" * 40)
        results['temporal_analysis'] = self._test_temporal_consistency()
        
        # Step 6: Data leakage detection
        print("\n🚨 STEP 6: Data Leakage Detection")
        print("-" * 40)
        results['leakage_analysis'] = self._test_data_leakage()
        
        # Step 7: Generate final diagnosis
        print("\n🏥 FINAL DIAGNOSIS")
        print("-" * 40)
        results['diagnosis'] = self._generate_diagnosis(results)
        
        return results
    
    def _test_data_loading(self) -> Dict[str, Any]:
        """Test data loading and basic validation"""
        
        try:
            # Load model
            model_path = self.model_folder / 'model.pkl'
            if not model_path.exists():
                return {'status': 'FAILED', 'error': f'Model not found: {model_path}'}
            
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded: {type(self.model).__name__}")
            
            # Load data
            if not self.data_file.exists():
                return {'status': 'FAILED', 'error': f'Data file not found: {self.data_file}'}
            
            self.data = pd.read_csv(self.data_file)
            print(f"✅ Data loaded: {len(self.data):,} rows, {len(self.data.columns)} columns")
            
            # Generate features using same logic as backtesting
            feature_generator = ModelAwareFeatureGenerator()
            self.data = feature_generator.generate_features(self.data, 'spread_compression_patterns_v3')
            print(f"✅ Features generated: {len(self.data.columns)} total columns")
            
            # Generate targets
            self.data = spread_compression_patterns_v3(self.data)
            print(f"✅ Targets generated")
            
            # Validate feature columns match model expectations
            expected_features = ['volume', 'compression_prediction_strength', 'microstructure_health',
                               'volume_ratio', 'spread_baseline', 'volatility_short']
            
            missing_features = [f for f in expected_features if f not in self.data.columns]
            if missing_features:
                return {'status': 'FAILED', 'error': f'Missing features: {missing_features}'}
            
            self.features = self.data[expected_features].copy()
            self.targets = self.data['target'].copy()
            
            print(f"✅ Feature validation passed: {len(expected_features)} features")
            print(f"✅ Target validation passed: {len(self.targets)} targets")
            
            return {
                'status': 'SUCCESS',
                'data_rows': len(self.data),
                'feature_columns': len(expected_features),
                'target_distribution': self.targets.value_counts().to_dict()
            }
            
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}
    
    def _test_model_predictions(self) -> Dict[str, Any]:
        """Test model prediction generation"""
        
        try:
            # Clean data for predictions
            clean_features = self.features.dropna()
            clean_indices = clean_features.index
            
            print(f"📊 Clean data: {len(clean_features):,} rows (dropped {len(self.features) - len(clean_features):,} NaN)")
            
            # Generate predictions
            self.predictions = self.model.predict(clean_features)
            self.probabilities = self.model.predict_proba(clean_features)
            
            print(f"✅ Predictions generated: {len(self.predictions):,}")
            print(f"✅ Probabilities generated: {self.probabilities.shape}")
            
            # Analyze prediction distribution
            pred_dist = np.bincount(self.predictions.astype(int))
            prob_stats = {
                'min_confidence': float(np.min(np.max(self.probabilities, axis=1))),
                'max_confidence': float(np.max(np.max(self.probabilities, axis=1))),
                'mean_confidence': float(np.mean(np.max(self.probabilities, axis=1))),
                'unique_values': len(np.unique(np.max(self.probabilities, axis=1)))
            }
            
            print(f"📈 Prediction distribution: {dict(enumerate(pred_dist))}")
            print(f"📈 Confidence stats: {prob_stats}")
            
            # Store clean indices for later analysis
            self.clean_indices = clean_indices
            
            return {
                'status': 'SUCCESS',
                'prediction_count': len(self.predictions),
                'prediction_distribution': dict(enumerate(pred_dist)),
                'confidence_stats': prob_stats,
                'clean_data_pct': len(clean_features) / len(self.features)
            }
            
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}
    
    def _test_directional_accuracy(self) -> Dict[str, Any]:
        """Test if predictions match actual price movements"""
        
        try:
            # Get clean data subset
            clean_data = self.data.loc[self.clean_indices].copy()
            
            # Calculate actual future returns at different horizons
            horizons = [1, 3, 5, 10, 15, 30]  # minutes
            
            directional_accuracy = {}
            
            for horizon in horizons:
                # Calculate future return
                future_return = clean_data['close'].shift(-horizon) / clean_data['close'] - 1
                
                # Convert to directional signal (1=buy, 2=sell, 0=hold)
                actual_direction = np.zeros(len(future_return))
                actual_direction[future_return > 0.001] = 1  # BUY if >0.1% move up
                actual_direction[future_return < -0.001] = 2  # SELL if >0.1% move down
                
                # Compare with predictions
                valid_mask = (self.predictions != 0) & (~np.isnan(future_return))
                
                if np.sum(valid_mask) == 0:
                    directional_accuracy[f'{horizon}min'] = {'accuracy': 0, 'samples': 0}
                    continue
                
                pred_subset = self.predictions[valid_mask]
                actual_subset = actual_direction[valid_mask]
                
                # Calculate accuracy for trading signals only (exclude HOLD)
                trading_mask = (pred_subset != 0) & (actual_subset != 0)
                
                if np.sum(trading_mask) == 0:
                    accuracy = 0
                    samples = 0
                else:
                    accuracy = np.mean(pred_subset[trading_mask] == actual_subset[trading_mask])
                    samples = np.sum(trading_mask)
                
                directional_accuracy[f'{horizon}min'] = {
                    'accuracy': float(accuracy),
                    'samples': int(samples),
                    'pred_buy': int(np.sum(pred_subset == 1)),
                    'pred_sell': int(np.sum(pred_subset == 2)),
                    'actual_up': int(np.sum(actual_subset == 1)),
                    'actual_down': int(np.sum(actual_subset == 2))
                }
                
                print(f"  {horizon:2d}min: {accuracy*100:5.1f}% accuracy ({samples:,} samples)")
            
            # Best horizon analysis
            best_horizon = max(directional_accuracy.keys(), 
                             key=lambda x: directional_accuracy[x]['accuracy'])
            best_accuracy = directional_accuracy[best_horizon]['accuracy']
            
            print(f"\n🎯 Best horizon: {best_horizon} with {best_accuracy*100:.1f}% accuracy")
            
            if best_accuracy < 0.4:
                print("❌ CRITICAL: Model predictions are worse than random!")
            elif best_accuracy < 0.5:
                print("⚠️  WARNING: Model predictions are below random (50%)")
            else:
                print("✅ Model shows directional predictive power")
            
            return {
                'status': 'SUCCESS',
                'directional_accuracy': directional_accuracy,
                'best_horizon': best_horizon,
                'best_accuracy': best_accuracy
            }
            
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}
    
    def _test_feature_target_relationships(self) -> Dict[str, Any]:
        """Test feature-target correlations"""
        
        try:
            clean_data = self.data.loc[self.clean_indices].copy()
            
            # Calculate correlations between features and actual targets
            feature_cols = ['volume', 'compression_prediction_strength', 'microstructure_health',
                           'volume_ratio', 'spread_baseline', 'volatility_short']
            
            correlations = {}
            for feature in feature_cols:
                if feature in clean_data.columns:
                    corr = clean_data[feature].corr(clean_data['target'])
                    correlations[feature] = float(corr) if not np.isnan(corr) else 0.0
                    print(f"  {feature}: {corr:.4f}")
            
            # Test target distribution in backtesting period
            target_dist = clean_data['target'].value_counts(normalize=True).to_dict()
            print(f"\n📊 Target distribution: {target_dist}")
            
            # Check if model learned meaningful patterns
            avg_correlation = np.mean([abs(c) for c in correlations.values()])
            print(f"\n📈 Average absolute correlation: {avg_correlation:.4f}")
            
            if avg_correlation < 0.1:
                print("❌ CRITICAL: Features have very weak correlation with targets!")
            elif avg_correlation < 0.2:
                print("⚠️  WARNING: Features have weak correlation with targets")
            else:
                print("✅ Features show meaningful correlation with targets")
            
            return {
                'status': 'SUCCESS',
                'feature_correlations': correlations,
                'target_distribution': target_dist,
                'avg_correlation': avg_correlation
            }
            
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}
    
    def _test_temporal_consistency(self) -> Dict[str, Any]:
        """Test if model performance is consistent across time periods"""
        
        try:
            clean_data = self.data.loc[self.clean_indices].copy()
            
            # Split data into time periods
            clean_data = clean_data.sort_values('timestamp')
            n_periods = 5
            period_size = len(clean_data) // n_periods
            
            temporal_results = {}
            
            for i in range(n_periods):
                start_idx = i * period_size
                end_idx = (i + 1) * period_size if i < n_periods - 1 else len(clean_data)
                
                period_data = clean_data.iloc[start_idx:end_idx]
                period_predictions = self.predictions[start_idx:end_idx]
                
                # Calculate confidence distribution for this period
                period_probs = self.probabilities[start_idx:end_idx]
                period_confidence = np.max(period_probs, axis=1)
                
                temporal_results[f'period_{i+1}'] = {
                    'start_date': str(period_data['timestamp'].iloc[0])[:10],
                    'end_date': str(period_data['timestamp'].iloc[-1])[:10],
                    'samples': len(period_data),
                    'pred_distribution': dict(np.bincount(period_predictions.astype(int))),
                    'mean_confidence': float(np.mean(period_confidence)),
                    'trading_signals': int(np.sum(period_predictions != 0))
                }
                
                print(f"  Period {i+1}: {temporal_results[f'period_{i+1}']['start_date']} to "
                      f"{temporal_results[f'period_{i+1}']['end_date']} - "
                      f"{temporal_results[f'period_{i+1}']['trading_signals']} signals")
            
            return {
                'status': 'SUCCESS',
                'temporal_analysis': temporal_results
            }
            
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}
    
    def _test_data_leakage(self) -> Dict[str, Any]:
        """Test for potential data leakage"""
        
        try:
            # Test 1: Check if features contain future information
            clean_data = self.data.loc[self.clean_indices].copy()
            
            leakage_tests = {}
            
            # Test correlation between features and future returns
            future_return_1min = clean_data['close'].shift(-1) / clean_data['close'] - 1
            future_return_5min = clean_data['close'].shift(-5) / clean_data['close'] - 1
            
            feature_cols = ['volume', 'compression_prediction_strength', 'microstructure_health',
                           'volume_ratio', 'spread_baseline', 'volatility_short']
            
            future_correlations = {}
            for feature in feature_cols:
                if feature in clean_data.columns:
                    corr_1min = clean_data[feature].corr(future_return_1min)
                    corr_5min = clean_data[feature].corr(future_return_5min)
                    
                    future_correlations[feature] = {
                        '1min_future_corr': float(corr_1min) if not np.isnan(corr_1min) else 0.0,
                        '5min_future_corr': float(corr_5min) if not np.isnan(corr_5min) else 0.0
                    }
            
            # Check for suspiciously high correlations with future returns
            max_future_corr = max([
                max(abs(v['1min_future_corr']), abs(v['5min_future_corr'])) 
                for v in future_correlations.values()
            ])
            
            print(f"📊 Maximum future correlation: {max_future_corr:.4f}")
            
            if max_future_corr > 0.3:
                print("🚨 CRITICAL: Potential data leakage detected!")
                leakage_risk = "HIGH"
            elif max_future_corr > 0.15:
                print("⚠️  WARNING: Moderate correlation with future returns")
                leakage_risk = "MEDIUM"
            else:
                print("✅ No significant data leakage detected")
                leakage_risk = "LOW"
            
            return {
                'status': 'SUCCESS',
                'future_correlations': future_correlations,
                'max_future_correlation': max_future_corr,
                'leakage_risk': leakage_risk
            }
            
        except Exception as e:
            return {'status': 'FAILED', 'error': str(e)}
    
    def _generate_diagnosis(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate final diagnosis based on all test results"""
        
        diagnosis = {
            'primary_issue': 'UNKNOWN',
            'confidence': 'LOW',
            'recommended_action': 'INVESTIGATE_FURTHER',
            'summary': 'Diagnostic incomplete'
        }
        
        try:
            # Analyze directional accuracy
            if 'directional_accuracy' in results and results['directional_accuracy']['status'] == 'SUCCESS':
                best_accuracy = results['directional_accuracy']['best_accuracy']
                
                if best_accuracy < 0.4:
                    diagnosis.update({
                        'primary_issue': 'INVERSE_PREDICTIONS',
                        'confidence': 'HIGH',
                        'recommended_action': 'INVERT_SIGNALS_OR_RETRAIN',
                        'summary': f'Model predictions are consistently wrong direction ({best_accuracy*100:.1f}% accuracy). Consider inverting signals or retraining model.'
                    })
                elif best_accuracy < 0.5:
                    diagnosis.update({
                        'primary_issue': 'POOR_DIRECTIONAL_ACCURACY',
                        'confidence': 'HIGH',
                        'recommended_action': 'RETRAIN_MODEL',
                        'summary': f'Model has poor directional accuracy ({best_accuracy*100:.1f}%). Model needs retraining with better features or different approach.'
                    })
            
            # Check feature-target relationships
            if 'feature_target_analysis' in results and results['feature_target_analysis']['status'] == 'SUCCESS':
                avg_corr = results['feature_target_analysis']['avg_correlation']
                
                if avg_corr < 0.1 and diagnosis['primary_issue'] == 'UNKNOWN':
                    diagnosis.update({
                        'primary_issue': 'WEAK_FEATURE_TARGET_RELATIONSHIP',
                        'confidence': 'HIGH',
                        'recommended_action': 'IMPROVE_FEATURES_OR_TARGETS',
                        'summary': f'Features have very weak correlation with targets ({avg_corr:.3f}). Need better feature engineering or target definition.'
                    })
            
            # Check for data leakage
            if 'leakage_analysis' in results and results['leakage_analysis']['status'] == 'SUCCESS':
                if results['leakage_analysis']['leakage_risk'] == 'HIGH':
                    diagnosis.update({
                        'primary_issue': 'DATA_LEAKAGE',
                        'confidence': 'HIGH',
                        'recommended_action': 'FIX_DATA_LEAKAGE',
                        'summary': 'Data leakage detected. Features contain future information. Fix feature generation pipeline.'
                    })
            
            print(f"🏥 PRIMARY ISSUE: {diagnosis['primary_issue']}")
            print(f"🎯 CONFIDENCE: {diagnosis['confidence']}")
            print(f"💡 RECOMMENDED ACTION: {diagnosis['recommended_action']}")
            print(f"📝 SUMMARY: {diagnosis['summary']}")
            
        except Exception as e:
            diagnosis['summary'] = f'Diagnosis failed: {str(e)}'
        
        return diagnosis

def run_model_diagnostic(model_folder: str, data_file: str):
    """Run complete model diagnostic"""
    
    framework = ModelDiagnosticFramework(model_folder, data_file)
    results = framework.run_full_diagnostic()
    
    # Save results
    import json
    output_file = Path("model_diagnostic_results.json")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_types(results)
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    # Run diagnostic on your model
    model_folder = "storage/models/2025-07-30_19:30:29"
    data_file = "storage/data/capital_com/GBPUSD/processed/1m/2025-03-02_2025-07-01.csv"
    
    results = run_model_diagnostic(model_folder, data_file)