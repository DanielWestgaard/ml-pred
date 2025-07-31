"""
Target Validation Analysis System
================================

Systematic validation of target generation logic to identify if targets
correctly capture profitable spread compression patterns.

This validates:
1. Target-feature correlation strength
2. Target distribution quality  
3. Signal timing accuracy
4. Forward-looking profit validation (without data leakage)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

from data.processing.cleaning import DataCleaner
from data.processing.validation import DataValidator
warnings.filterwarnings('ignore')

from data.features.enhanced_generation import ModelAwareFeatureGenerator
from data.features.target_hypothesis import TargetHypothesis
from data.features.feature_importance import FeatureImportanceAnalyzer
from utils import data_utils
from utils.logging_utils import get_logger

logger = get_logger()

class TargetValidationAnalyzer:
    """
    Comprehensive target validation system to ensure targets capture
    profitable spread compression patterns correctly.
    """
    
    def __init__(self, dataset_path: str, target_hypothesis: str = 'spread_compression_patterns_v3'):
        """
        Initialize target validation analyzer
        
        Args:
            dataset_path: Path to base dataset (without features/targets)
            target_hypothesis: Target generation method to validate
        """
        self.dataset_path = dataset_path
        self.target_hypothesis = target_hypothesis
        self.base_df = None
        self.features_df = None
        self.target_df = None
        self.validation_results = {}
        
        logger.info(f"Initializing target validation for hypothesis: {target_hypothesis}")
    
    def _generate_features_with_dependencies(self, feature_generator):
        """
        Generate features with proper dependency handling, similar to target_hypothesis.py V3
        """
        # Define feature dependency mapping
        feature_dependency_map = {
            'volume_ma_20': ('volume_ma_20_feature', []),
            'volume_ratio': ('volume_ratio_feature', ['volume_ma_20']),  
            'depth_stability': ('enhanced_market_depth_features', []),
            'liquidity_stress': ('enhanced_market_depth_features', []),
            'institutional_footprint': ('institutional_footprint_detection', ['volume_ratio']),
            'microstructure_health': ('enhanced_microstructure_health', []),
            'signal_conviction': ('enhanced_predictive_signals', []),
            'london_session': ('fx_trading_sessions', [])
        }
        
        # Required features for analysis
        required_features = [
            'depth_stability', 'liquidity_stress',
            'institutional_footprint', 'microstructure_health', 
            'signal_conviction', 'london_session', 'volume_ratio'
        ]
        
        # Track original columns to identify what's added
        original_columns = set(feature_generator.df.columns)
        
        # Build complete dependency graph including transitive dependencies
        all_needed_features = set(required_features)
        
        def add_dependencies_recursively(feature_name):
            if feature_name in feature_dependency_map:
                _, dependencies = feature_dependency_map[feature_name]
                if dependencies:
                    for dep in dependencies:
                        if dep not in feature_generator.df.columns:
                            all_needed_features.add(dep)
                            add_dependencies_recursively(dep)
        
        for feature in required_features:
            add_dependencies_recursively(feature)
        
        # Topologically sort features based on dependencies
        def topological_sort():
            visited = set()
            temp_visited = set()
            result = []
            
            def visit(feature):
                if feature in temp_visited:
                    logger.warning(f"Circular dependency detected involving {feature}")
                    return
                if feature in visited:
                    return
                    
                temp_visited.add(feature)
                
                if feature in feature_dependency_map:
                    _, dependencies = feature_dependency_map[feature]
                    if dependencies:
                        for dep in dependencies:
                            if dep in all_needed_features:
                                visit(dep)
                
                temp_visited.remove(feature)
                visited.add(feature)
                result.append(feature)
            
            for feature in all_needed_features:
                if feature not in visited:
                    visit(feature)
            
            return result
        
        generation_order = topological_sort()
        successfully_generated = []
        
        # Generate features in dependency order
        for feature_name in generation_order:
            if feature_name not in feature_generator.df.columns:
                if feature_name in feature_dependency_map:
                    generator_method, _ = feature_dependency_map[feature_name]
                    if generator_method and hasattr(feature_generator, generator_method):
                        try:
                            logger.debug(f"Generating {feature_name} using {generator_method}")
                            feature_generator.__getattribute__(generator_method)()
                            if feature_name in feature_generator.df.columns:
                                successfully_generated.append(feature_name)
                                logger.debug(f"Successfully generated {feature_name}")
                            else:
                                logger.warning(f"Feature {feature_name} not found after generation")
                        except Exception as e:
                            logger.warning(f"Failed to generate {feature_name}: {e}")
                    else:
                        logger.warning(f"Cannot generate missing feature: {feature_name} (no method {generator_method})")
        
        logger.info(f"Successfully generated {len(successfully_generated)} features: {successfully_generated}")
        return successfully_generated
    
    def load_and_prepare_data(self) -> bool:
        """Load base data and prepare for analysis"""
        try:
            # Load base dataset
            self.base_df, _ = data_utils.check_and_return_df(self.dataset_path)
            logger.info(f"Loaded base dataset: {len(self.base_df)} rows, {len(self.base_df.columns)} columns")
            
            # Clean first
            self.cleaner = DataCleaner(self.base_df, save_cleaned_data=False)
            self.df = self.cleaner.run()
            
            # Validate
            self.validator = DataValidator(self.df)
            self.validator_results = self.validator.run()
            self.df = self.validator_results["validated_data"]
            # logger results
            data_utils.check_validation(self.validator_results["is_valid"], self.validator_results["issues"])
            
            # Generate features using enhanced generation
            logger.info("Generating enhanced features...")
            feature_generator = ModelAwareFeatureGenerator(self.df)
            
            # Generate key feature sets for spread compression analysis with dependency handling
            self._generate_features_with_dependencies(feature_generator)
            
            self.features_df = feature_generator.df.copy()
            logger.info(f"Generated features: {len(self.features_df.columns)} total columns")
            
            # Generate target using specified hypothesis
            logger.info(f"Generating target using hypothesis: {self.target_hypothesis}")
            target_generator = TargetHypothesis(self.features_df, self.target_hypothesis)
            self.target_df = target_generator.run()
            
            # Identify target columns
            target_columns = [col for col in self.target_df.columns if col.startswith('target_')]
            logger.info(f"Generated target columns: {target_columns}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load and prepare data: {e}")
            return False
    
    def validate_target_distribution(self) -> Dict:
        """Analyze target distribution for quality assessment"""
        logger.info("=== TARGET DISTRIBUTION VALIDATION ===")
        
        results = {}
        target_columns = [col for col in self.target_df.columns if col.startswith('target_')]
        
        for target_col in target_columns:
            target_series = self.target_df[target_col]
            
            # Basic distribution stats
            distribution = target_series.value_counts().sort_index()
            total_samples = len(target_series)
            
            results[target_col] = {
                'distribution': distribution.to_dict(),
                'total_samples': total_samples,
                'class_percentages': (distribution / total_samples * 100).to_dict(),
                'balance_quality': self._assess_balance_quality(distribution),
                'signal_rate': (target_series > 0).mean() if target_series.dtype in ['int64', 'float64'] else None
            }
            
            logger.info(f"\n{target_col}:")
            logger.info(f"  Distribution: {distribution.to_dict()}")
            for class_val, count in distribution.items():
                pct = count / total_samples * 100
                logger.info(f"    Class {class_val}: {count:,} samples ({pct:.2f}%)")
        
        self.validation_results['target_distribution'] = results
        return results
    
    def _assess_balance_quality(self, distribution: pd.Series) -> str:
        """Assess if class distribution is suitable for ML training"""
        percentages = distribution / distribution.sum() * 100
        
        # Check for severe imbalance (>95% in one class)
        if percentages.max() > 95:
            return "SEVERE_IMBALANCE"
        elif percentages.max() > 85:
            return "MODERATE_IMBALANCE"  
        elif percentages.max() > 70:
            return "MILD_IMBALANCE"
        else:
            return "BALANCED"
    
    def analyze_feature_target_correlation(self) -> Dict:
        """Analyze correlation between features and targets"""
        logger.info("=== FEATURE-TARGET CORRELATION ANALYSIS ===")
        
        results = {}
        target_columns = [col for col in self.target_df.columns if col.startswith('target_')]
        
        for target_col in target_columns:
            logger.info(f"\nAnalyzing correlations for {target_col}...")
            
            # Use feature importance analyzer for comprehensive analysis
            importance_analyzer = FeatureImportanceAnalyzer(
                self.target_df, 
                target_column=target_col,
                task_type='classification' if self.target_df[target_col].nunique() <= 10 else 'regression'
            )
            
            # Run all importance methods
            importance_results = importance_analyzer.analyze_all_methods()
            consensus_ranking = importance_analyzer.get_consensus_ranking(top_n=20)
            
            results[target_col] = {
                'importance_methods': importance_results,
                'consensus_top_20': consensus_ranking.to_dict(),
                'top_correlations': importance_results.get('correlation', pd.Series()).head(10).to_dict()
            }
            
            # Log top correlations
            logger.info(f"Top 10 correlated features for {target_col}:")
            for i, (feature, score) in enumerate(consensus_ranking.head(10).items(), 1):
                logger.info(f"  {i:2d}. {feature:<35} {score:.4f}")
        
        self.validation_results['feature_correlation'] = results
        return results
    
    def validate_signal_timing(self) -> Dict:
        """Analyze timing patterns of generated signals"""
        logger.info("=== SIGNAL TIMING VALIDATION ===")
        
        results = {}
        target_columns = [col for col in self.target_df.columns if col.startswith('target_')]
        
        # Ensure we have datetime index for timing analysis
        if 'date' in self.target_df.columns:
            df_with_time = self.target_df.copy()
            df_with_time['datetime'] = pd.to_datetime(df_with_time['date'])
            df_with_time['hour'] = df_with_time['datetime'].dt.hour
            df_with_time['day_of_week'] = df_with_time['datetime'].dt.dayofweek
        else:
            logger.warning("No date column found - skipping timing analysis")
            return results
        
        for target_col in target_columns:
            if df_with_time[target_col].dtype not in ['int64', 'float64']:
                continue
                
            signals = df_with_time[df_with_time[target_col] > 0] if df_with_time[target_col].nunique() > 2 else df_with_time[df_with_time[target_col] == 1]
            
            if len(signals) == 0:
                logger.warning(f"No signals found for {target_col}")
                continue
            
            # Hourly distribution
            hourly_dist = signals['hour'].value_counts().sort_index()
            
            # Daily distribution  
            daily_dist = signals['day_of_week'].value_counts().sort_index()
            
            # Session analysis (if london_session exists)
            session_analysis = {}
            if 'london_session' in df_with_time.columns:
                london_signals = signals[signals['london_session'] == 1]
                session_analysis = {
                    'total_signals': len(signals),
                    'london_signals': len(london_signals),
                    'london_signal_rate': len(london_signals) / len(signals) * 100 if len(signals) > 0 else 0
                }
            
            results[target_col] = {
                'total_signals': len(signals),
                'hourly_distribution': hourly_dist.to_dict(),
                'daily_distribution': daily_dist.to_dict(),
                'session_analysis': session_analysis,
                'signal_density': len(signals) / len(df_with_time) * 100
            }
            
            logger.info(f"\n{target_col} Timing Analysis:")
            logger.info(f"  Total signals: {len(signals):,}")
            logger.info(f"  Signal density: {len(signals) / len(df_with_time) * 100:.2f}%")
            logger.info(f"  Peak hours: {hourly_dist.nlargest(3).to_dict()}")
            
            if session_analysis:
                logger.info(f"  London session signals: {session_analysis['london_signal_rate']:.1f}%")
        
        self.validation_results['signal_timing'] = results
        return results
    
    def forward_profit_validation(self, forward_periods: List[int] = [5, 10, 30]) -> Dict:
        """
        Validate if signals lead to profitable outcomes without data leakage.
        This analyzes future returns for signals generated using only past data.
        """
        logger.info("=== FORWARD PROFIT VALIDATION ===")
        
        results = {}
        target_columns = [col for col in self.target_df.columns if col.startswith('target_')]
        
        # Ensure we have price data
        if 'close' not in self.target_df.columns:
            logger.error("No 'close' price column found - cannot validate profitability")
            return results
        
        df_profit = self.target_df.copy()
        
        for target_col in target_columns:
            logger.info(f"\nValidating profitability for {target_col}...")
            
            target_results = {}
            
            for periods in forward_periods:
                # Calculate forward returns (this is validation, not training)
                df_profit[f'forward_return_{periods}'] = (
                    df_profit['close'].shift(-periods) / df_profit['close'] - 1
                ) * 100
                
                # Analyze returns by signal class
                if df_profit[target_col].dtype in ['int64', 'float64']:
                    unique_classes = sorted(df_profit[target_col].unique())
                else:
                    continue
                
                class_performance = {}
                for class_val in unique_classes:
                    mask = df_profit[target_col] == class_val
                    class_returns = df_profit.loc[mask, f'forward_return_{periods}'].dropna()
                    
                    if len(class_returns) > 0:
                        class_performance[f'class_{class_val}'] = {
                            'mean_return': class_returns.mean(),
                            'median_return': class_returns.median(),
                            'std_return': class_returns.std(),
                            'win_rate': (class_returns > 0).mean() * 100,
                            'sample_count': len(class_returns),
                            'total_return': class_returns.sum()
                        }
                
                target_results[f'{periods}_periods'] = class_performance
                
                # Log performance summary
                logger.info(f"  {periods}-period forward returns:")
                for class_name, perf in class_performance.items():
                    logger.info(f"    {class_name}: {perf['mean_return']:.4f}% avg, "
                              f"{perf['win_rate']:.1f}% win rate ({perf['sample_count']} samples)")
            
            results[target_col] = target_results
        
        self.validation_results['profit_validation'] = results
        return results
    
    def create_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        if not self.validation_results:
            return "No validation results available. Run validation methods first."
        
        report = []
        report.append("=" * 80)
        report.append("TARGET VALIDATION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Dataset: {self.dataset_path}")
        report.append(f"Target Hypothesis: {self.target_hypothesis}")
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Distribution Analysis
        if 'target_distribution' in self.validation_results:
            report.append("TARGET DISTRIBUTION SUMMARY:")
            report.append("-" * 40)
            for target_col, stats in self.validation_results['target_distribution'].items():
                report.append(f"\n{target_col}:")
                for class_val, pct in stats['class_percentages'].items():
                    report.append(f"  Class {class_val}: {pct:.2f}%")
                report.append(f"  Balance Quality: {stats['balance_quality']}")
                if stats['signal_rate']:
                    report.append(f"  Signal Rate: {stats['signal_rate']*100:.2f}%")
        
        # Feature Correlation
        if 'feature_correlation' in self.validation_results:
            report.append("\n\nTOP PREDICTIVE FEATURES:")
            report.append("-" * 40)
            for target_col, analysis in self.validation_results['feature_correlation'].items():
                report.append(f"\n{target_col} - Top 10 Features:")
                for i, (feature, score) in enumerate(analysis['consensus_top_20'].items(), 1):
                    if i <= 10:
                        report.append(f"  {i:2d}. {feature:<30} {score:.4f}")
        
        # Profit Validation
        if 'profit_validation' in self.validation_results:
            report.append("\n\nPROFITABILITY VALIDATION:")
            report.append("-" * 40)
            for target_col, periods_data in self.validation_results['profit_validation'].items():
                report.append(f"\n{target_col}:")
                for period, class_data in periods_data.items():
                    report.append(f"  {period}:")
                    for class_name, perf in class_data.items():
                        report.append(f"    {class_name}: {perf['mean_return']:.4f}% avg return, "
                                    f"{perf['win_rate']:.1f}% win rate")
        
        return "\n".join(report)
    
    def run_complete_validation(self, save_results: bool = True) -> Dict:
        """Run complete target validation analysis"""
        logger.info("Starting complete target validation analysis...")
        
        # Load and prepare data
        if not self.load_and_prepare_data():
            logger.error("Failed to load data - aborting validation")
            return {}
        
        # Run all validation methods
        self.validate_target_distribution()
        self.analyze_feature_target_correlation()
        self.validate_signal_timing()
        self.forward_profit_validation()
        
        # Generate report
        report = self.create_validation_report()
        logger.info("\n" + report)
        
        # Save results if requested
        if save_results:
            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = Path(f"storage/target_validation/{self.target_hypothesis}_{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save report
            with open(output_dir / "validation_report.txt", 'w') as f:
                f.write(report)
            
            # Save detailed results
            import json
            with open(output_dir / "detailed_results.json", 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                serializable_results = self._make_json_serializable(self.validation_results)
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Validation results saved to: {output_dir}")
        
        return self.validation_results
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

# Convenience function for quick analysis
def validate_target_hypothesis(dataset_path: str, hypothesis: str = 'spread_compression_patterns_v3'):
    """
    Quick target validation for a specific hypothesis
    
    Args:
        dataset_path: Path to base dataset
        hypothesis: Target hypothesis to validate
    
    Returns:
        Validation results dictionary
    """
    validator = TargetValidationAnalyzer(dataset_path, hypothesis)
    return validator.run_complete_validation()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        hypothesis = sys.argv[2] if len(sys.argv) > 2 else 'spread_compression_patterns_v3'
        
        print(f"Validating target hypothesis '{hypothesis}' on dataset: {data_path}")
        results = validate_target_hypothesis(data_path, hypothesis)
        print("\nValidation completed!")
    else:
        print("Usage: python target_validation_analysis.py <data_path> [hypothesis_name]")