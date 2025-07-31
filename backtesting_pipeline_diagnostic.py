#!/usr/bin/env python3
"""
Backtesting Pipeline Diagnostic Tool

Systematically traces the signal flow through the entire backtesting pipeline
to identify where the 408,774 model predictions become only 2-3 trades.

Usage:
    python backtesting_pipeline_diagnostic.py <config_path>
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logging_utils import get_logger
from utils.hypothesis.config_loader import load_hypothesis_config
from utils.model.model_metadata_loader import load_model_metadata
from utils.model.model_type_detector import ModelTypeDetector, PredictionProcessor
from data.features.enhanced_generation import ModelAwareFeatureGenerator
from strategy.components.signal_processor import SignalProcessor, AdvancedSignalProcessor
from backtesting.backtest_data_manager import BacktestDataManager

logger = get_logger()

class BacktestingPipelineDiagnostic:
    """Comprehensive diagnostic for backtesting pipeline."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None
        self.model = None
        self.model_metadata = None
        self.model_type = None
        self.data = None
        
        # Pipeline components
        self.prediction_processor = None
        self.signal_processor = None
        
        # Tracking
        self.pipeline_stats = {
            'total_data_points': 0,
            'valid_data_points': 0,
            'missing_features': 0,
            'predictions_generated': 0,
            'raw_signals': 0,
            'filtered_signals': 0,
            'confidence_filtered': 0,
            'advanced_filtered': 0,
            'final_trades': 0
        }
        
    def run_full_diagnostic(self):
        """Run complete diagnostic of the backtesting pipeline."""
        logger.info("🔍 Starting comprehensive backtesting pipeline diagnostic...")
        
        try:
            self._load_configuration()
            self._load_model_and_metadata()
            self._load_and_analyze_data()
            self._initialize_processors()
            self._run_pipeline_trace()
            self._generate_diagnostic_report()
            
            logger.info("✅ Diagnostic completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Diagnostic failed: {e}")
            return False
    
    def _load_configuration(self):
        """Load configuration from hypothesis config file."""
        logger.info("📋 Loading configuration...")
        config_loader = load_hypothesis_config(self.config_path)
        self.config = config_loader.handle_hypothesis_source()
        
        backtest_config = self.config.get('backtesting', {})
        logger.info(f"Backtesting enabled: {backtest_config.get('enabled', False)}")
        logger.info(f"Model folder: {backtest_config.get('model_folder')}")
        logger.info(f"Data source: {backtest_config.get('data_source')}")
        logger.info(f"Date range: {backtest_config.get('start_date')} to {backtest_config.get('end_date')}")
        
    def _load_model_and_metadata(self):
        """Load trained model and metadata."""
        logger.info("🤖 Loading model and metadata...")
        
        backtest_config = self.config.get('backtesting', {})
        model_folder = backtest_config.get('model_folder')
        
        if not model_folder:
            raise ValueError("No model folder specified in config")
        
        # Load model
        model_files = list(Path(model_folder).glob('*.pkl'))
        if not model_files:
            raise ValueError(f"No .pkl files found in {model_folder}")
        
        model_path = model_files[0]
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info(f"Model loaded: {type(self.model).__name__}")
        
        # Load metadata
        metadata_path = Path(model_folder) / 'model_metadata.yml'
        if metadata_path.exists():
            import yaml
            with open(metadata_path, 'r') as f:
                self.model_metadata = yaml.safe_load(f)
            logger.info("Model metadata loaded successfully")
        else:
            logger.warning("No model metadata found")
            
        # Detect model type
        detector = ModelTypeDetector()
        self.model_type = detector.detect_from_model(self.model)
        logger.info(f"Model type detected: {self.model_type.value}")
        
    def _load_and_analyze_data(self):
        """Load and analyze backtesting data."""
        logger.info("📊 Loading and analyzing data...")
        
        backtest_config = self.config.get('backtesting', {})
        data_source = backtest_config.get('data_source')
        
        if not data_source or not os.path.exists(data_source):
            raise ValueError(f"Data source not found: {data_source}")
        
        # Load data
        self.data = pd.read_csv(data_source)
        logger.info(f"Raw data loaded: {len(self.data)} rows, {len(self.data.columns)} columns")
        
        # Filter by date range
        start_date = backtest_config.get('start_date')
        end_date = backtest_config.get('end_date')
        
        if start_date or end_date:
            self.data['timestamp'] = pd.to_datetime(self.data['date'], utc=True)
            original_size = len(self.data)
            
            if start_date:
                start_date = pd.to_datetime(start_date, utc=True)
                self.data = self.data[self.data['timestamp'] >= start_date]
                
            if end_date:
                end_date = pd.to_datetime(end_date, utc=True)
                self.data = self.data[self.data['timestamp'] <= end_date]
                
            logger.info(f"Date filtered: {original_size} → {len(self.data)} rows")
        
        self.pipeline_stats['total_data_points'] = len(self.data)
        
        # Check feature availability
        expected_features = self.model_metadata.get('dataset', {}).get('features', [])
        available_features = [f for f in expected_features if f in self.data.columns]
        missing_features = [f for f in expected_features if f not in self.data.columns]
        
        logger.info(f"Expected features: {len(expected_features)}")
        logger.info(f"Available features: {len(available_features)}")
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            
        # Check data quality
        logger.info(f"Data date range: {self.data['date'].min()} to {self.data['date'].max()}")
        
        if 'close_og' in self.data.columns:
            valid_prices = (self.data['close_og'] > 0).sum()
            logger.info(f"Valid price data: {valid_prices}/{len(self.data)} ({valid_prices/len(self.data)*100:.1f}%)")
        
    def _initialize_processors(self):
        """Initialize prediction and signal processors."""
        logger.info("⚙️ Initializing processors...")
        
        # Initialize prediction processor
        self.prediction_processor = PredictionProcessor(self.model_type)
        
        # Initialize signal processor
        signal_config = self.config.get('backtesting', {}).get('signal_processing', {})
        
        if signal_config.get('use_advanced_filtering', False):
            logger.info("Using AdvancedSignalProcessor")
            self.signal_processor = AdvancedSignalProcessor(
                confidence_threshold=signal_config.get('confidence_threshold', 0.6),
                min_confidence_delta=signal_config.get('min_confidence_delta', 0.01),
                cooldown_minutes=signal_config.get('cooldown_minutes', 15),
                max_signals_per_hour=signal_config.get('max_signals_per_hour', 4),
                confidence_window_minutes=signal_config.get('confidence_window_minutes', 30),
                model_type=self.model_type
            )
        else:
            logger.info("Using basic SignalProcessor")
            self.signal_processor = SignalProcessor(
                confidence_threshold=signal_config.get('confidence_threshold', 0.6),
                signal_filters=signal_config.get('signal_filters', []),
                model_type=self.model_type
            )
            
        logger.info(f"Signal processor confidence threshold: {signal_config.get('confidence_threshold', 0.6)}")
        
    def _run_pipeline_trace(self):
        """Trace signals through the entire pipeline."""
        logger.info("🔬 Tracing signals through pipeline...")
        
        expected_features = self.model_metadata.get('dataset', {}).get('features', [])
        
        # Sample analysis on first 1000 rows for detailed tracing
        sample_size = min(1000, len(self.data))
        sample_data = self.data.head(sample_size)
        
        logger.info(f"Running detailed trace on {sample_size} data points...")
        
        predictions_list = []
        probabilities_list = []
        raw_signals_list = []
        filtered_signals_list = []
        
        for idx, row in sample_data.iterrows():
            # Check for missing features
            missing_features = [f for f in expected_features if f not in row.index or pd.isna(row[f])]
            if missing_features:
                self.pipeline_stats['missing_features'] += 1
                continue
                
            # Check for valid price data
            if 'close_og' in row.index and (pd.isna(row['close_og']) or row['close_og'] <= 0):
                continue
                
            self.pipeline_stats['valid_data_points'] += 1
            
            # Extract features
            try:
                features = row[expected_features].to_frame().T
                
                # Generate predictions
                prediction_results = self.prediction_processor.process_predictions(self.model, features)
                predictions = prediction_results['predictions']
                
                if self.model_type.value == 'multiclass_classification':
                    probabilities = prediction_results.get('probabilities')
                else:
                    probabilities = prediction_results.get('positive_probabilities')
                
                self.pipeline_stats['predictions_generated'] += 1
                predictions_list.append(predictions[0] if len(predictions) > 0 else None)
                probabilities_list.append(probabilities[0] if probabilities is not None and len(probabilities) > 0 else None)
                
                # Process into signals
                signals = self.signal_processor.process(
                    predictions,
                    market_data=None,
                    model=self.model,
                    prediction_probabilities=probabilities
                )
                
                num_raw_signals = len(predictions)
                num_filtered_signals = len(signals['processed_signals'])
                
                self.pipeline_stats['raw_signals'] += num_raw_signals
                self.pipeline_stats['filtered_signals'] += num_filtered_signals
                
                raw_signals_list.extend(predictions)
                filtered_signals_list.extend(signals['processed_signals'])
                
                # Log first few for debugging
                if idx < 10:
                    logger.debug(f"Sample {idx}: pred={predictions[0] if len(predictions) > 0 else 'None'}, "
                               f"prob={probabilities[0] if probabilities is not None and len(probabilities) > 0 else 'None'}, "
                               f"signals={num_filtered_signals}")
                
            except Exception as e:
                logger.debug(f"Error processing row {idx}: {e}")
                continue
        
        # Analyze predictions and probabilities
        if predictions_list:
            predictions_array = np.array([p for p in predictions_list if p is not None])
            logger.info(f\"\\n📊 PREDICTION ANALYSIS (sample):\")\n            logger.info(f\"  Predictions generated: {len(predictions_array)}\")\n            if len(predictions_array) > 0:\n                unique_preds = np.unique(predictions_array)\n                logger.info(f\"  Unique predictions: {unique_preds}\")\n                for pred in unique_preds:\n                    count = np.sum(predictions_array == pred)\n                    pct = (count / len(predictions_array)) * 100\n                    logger.info(f\"    Class {pred}: {count} ({pct:.1f}%)\")\n        \n        if probabilities_list:\n            # Analyze probabilities for multiclass\n            valid_probs = [p for p in probabilities_list if p is not None]\n            if valid_probs and len(valid_probs) > 0:\n                logger.info(f\"\\n🎯 PROBABILITY ANALYSIS (sample):\")\n                if isinstance(valid_probs[0], (list, np.ndarray)) and len(valid_probs[0]) == 3:\n                    # 3-class probabilities\n                    probs_array = np.array(valid_probs)\n                    max_probs = np.max(probs_array, axis=1)\n                    \n                    logger.info(f\"  Max probability stats:\")\n                    logger.info(f\"    Min: {max_probs.min():.4f}\")\n                    logger.info(f\"    Max: {max_probs.max():.4f}\")\n                    logger.info(f\"    Mean: {max_probs.mean():.4f}\")\n                    \n                    # Check confidence thresholds\n                    threshold = self.signal_processor.confidence_threshold\n                    above_threshold = np.sum(max_probs > threshold)\n                    logger.info(f\"    Above {threshold}: {above_threshold}/{len(max_probs)} ({above_threshold/len(max_probs)*100:.1f}%)\")\n                    \n        # Analyze signal filtering\n        logger.info(f\"\\n🔍 SIGNAL FILTERING ANALYSIS:\")\n        logger.info(f\"  Raw signals: {self.pipeline_stats['raw_signals']}\")\n        logger.info(f\"  Filtered signals: {self.pipeline_stats['filtered_signals']}\")\n        if self.pipeline_stats['raw_signals'] > 0:\n            filter_rate = (1 - self.pipeline_stats['filtered_signals'] / self.pipeline_stats['raw_signals']) * 100\n            logger.info(f\"  Filter rate: {filter_rate:.1f}%\")\n            \n    def _generate_diagnostic_report(self):\n        \"\"\"Generate comprehensive diagnostic report.\"\"\"\n        logger.info(\"\\n\" + \"=\"*80)\n        logger.info(\"🎯 BACKTESTING PIPELINE DIAGNOSTIC REPORT\")\n        logger.info(\"=\"*80)\n        \n        # Data pipeline analysis\n        logger.info(f\"\\n📊 DATA PIPELINE:\")\n        logger.info(f\"  Total data points: {self.pipeline_stats['total_data_points']}\")\n        logger.info(f\"  Valid data points: {self.pipeline_stats['valid_data_points']}\")\n        logger.info(f\"  Missing features: {self.pipeline_stats['missing_features']}\")\n        \n        if self.pipeline_stats['total_data_points'] > 0:\n            valid_rate = (self.pipeline_stats['valid_data_points'] / self.pipeline_stats['total_data_points']) * 100\n            logger.info(f\"  Data validity rate: {valid_rate:.1f}%\")\n            \n        # Prediction pipeline analysis\n        logger.info(f\"\\n🤖 PREDICTION PIPELINE:\")\n        logger.info(f\"  Predictions generated: {self.pipeline_stats['predictions_generated']}\")\n        logger.info(f\"  Raw signals: {self.pipeline_stats['raw_signals']}\")\n        logger.info(f\"  Filtered signals: {self.pipeline_stats['filtered_signals']}\")\n        \n        # Issue identification\n        logger.info(f\"\\n🚨 ISSUE IDENTIFICATION:\")\n        \n        issues = []\n        \n        # Check data issues\n        if self.pipeline_stats['total_data_points'] < 1000:\n            issues.append(f\"⚠️ Very small dataset: {self.pipeline_stats['total_data_points']} points\")\n            \n        if self.pipeline_stats['valid_data_points'] < self.pipeline_stats['total_data_points'] * 0.5:\n            issues.append(f\"⚠️ High data loss: Only {self.pipeline_stats['valid_data_points']}/{self.pipeline_stats['total_data_points']} valid\")\n            \n        if self.pipeline_stats['missing_features'] > 0:\n            issues.append(f\"⚠️ Missing features in {self.pipeline_stats['missing_features']} data points\")\n            \n        # Check prediction issues\n        if self.pipeline_stats['predictions_generated'] == 0:\n            issues.append(\"❌ CRITICAL: No predictions generated\")\n            \n        if self.pipeline_stats['filtered_signals'] < self.pipeline_stats['raw_signals'] * 0.01:\n            issues.append(f\"❌ CRITICAL: Excessive signal filtering ({self.pipeline_stats['filtered_signals']}/{self.pipeline_stats['raw_signals']})\")\n            \n        if not issues:\n            logger.info(\"  ✅ No critical issues detected in sample\")\n        else:\n            for issue in issues:\n                logger.info(f\"  {issue}\")\n                \n        # Recommendations\n        logger.info(f\"\\n💡 RECOMMENDATIONS:\")\n        \n        if self.pipeline_stats['total_data_points'] < 1000:\n            logger.info(\"1. 📈 INCREASE DATE RANGE: Current range is too small for meaningful backtesting\")\n            \n        if self.pipeline_stats['filtered_signals'] < 10:\n            logger.info(\"2. ⚙️ ADJUST SIGNAL THRESHOLDS: Lower confidence_threshold or disable advanced filtering\")\n            \n        if self.pipeline_stats['missing_features'] > 0:\n            logger.info(\"3. 🔧 FIX FEATURE PIPELINE: Ensure backtesting data has all required features\")\n            \n        # Scaling analysis\n        if self.pipeline_stats['valid_data_points'] > 0:\n            extrapolated_signals = (self.pipeline_stats['filtered_signals'] / min(1000, self.pipeline_stats['valid_data_points'])) * self.pipeline_stats['total_data_points']\n            logger.info(f\"\\n📈 SCALING PROJECTION:\")\n            logger.info(f\"  If sample pattern holds across full dataset:\")\n            logger.info(f\"  Expected signals: ~{extrapolated_signals:.0f}\")\n            \n            if extrapolated_signals < 10:\n                logger.info(f\"  🚨 This explains why backtesting only generates 2-3 trades!\")\n            \ndef main():\n    import argparse\n    parser = argparse.ArgumentParser(description='Backtesting Pipeline Diagnostic')\n    parser.add_argument('config_path', help='Path to hypothesis config file')\n    args = parser.parse_args()\n    \n    if not os.path.exists(args.config_path):\n        logger.error(f\"Config file not found: {args.config_path}\")\n        return False\n        \n    diagnostic = BacktestingPipelineDiagnostic(args.config_path)\n    return diagnostic.run_full_diagnostic()\n\nif __name__ == \"__main__\":\n    success = main()\n    sys.exit(0 if success else 1)