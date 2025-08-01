"""
Feature Importance Analysis System
================================

Modular feature importance analysis that works with any feature set from ModelAwareFeatureGenerator.
Provides multiple importance techniques to identify the most predictive features.

Usage:
    # Standalone analysis
    python -c "from data.features.feature_importance import FeatureImportanceAnalyzer; FeatureImportanceAnalyzer('data.csv').analyze_and_save()"
    
    # Integrated in pipeline
    importance_analyzer = FeatureImportanceAnalyzer(processed_data)
    results = importance_analyzer.analyze_all_methods()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import LassoCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import warnings
from utils.performance_utils import safe_sample_for_analysis, memory_monitor, PerformanceContext
warnings.filterwarnings('ignore')

from utils import data_utils
from utils.logging_utils import get_logger

logger = get_logger()

class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis using multiple methods
    """
    
    def __init__(self, dataset, target_column: str = None, task_type: str = 'auto'):
        """
        Initialize feature importance analyzer
        
        Args:
            dataset: Input dataset (DataFrame or path)
            target_column: Target column name (auto-detected if None)
            task_type: 'regression', 'classification', or 'auto'
        """
        try:
            # Load and validate dataset
            self.df, _ = data_utils.check_and_return_df(dataset)
            
            # Auto-detect target column
            if target_column is None:
                target_column = self._detect_target_column()
            
            self.target_column = target_column
            
            # Separate features and target
            self.features = self._prepare_features()
            self.target = self.df[self.target_column]
            
            # Determine task type
            self.task_type = self._determine_task_type(task_type)
            
            # Results storage
            self.importance_results = {}
            
            logger.info(f"FeatureImportanceAnalyzer initialized: {len(self.features.columns)} features, "
                       f"task: {self.task_type}, target: {self.target_column}")
            
        except Exception as e:
            logger.error(f"Error initializing FeatureImportanceAnalyzer: {e}")
            raise
    
    def _detect_target_column(self) -> str:
        """Auto-detect target column"""
        
        # Look for common target patterns
        target_candidates = [col for col in self.df.columns if col.startswith('target_')]
        
        if target_candidates:
            # Use the first target column found
            return target_candidates[0]
        
        # Fallback to 'close' for price prediction
        if 'close' in self.df.columns:
            return 'close'
        
        raise ValueError("Could not auto-detect target column. Please specify target_column parameter.")
    
    def _prepare_features(self) -> pd.DataFrame:
        """Prepare feature matrix by excluding non-feature columns"""
        
        # Columns to exclude from feature analysis
        exclude_patterns = ['target_', 'timestamp', 'datetime', 'date', 'time']
        exclude_exact = ['open', 'high', 'low', 'close', 'volume', 'close_og']
        
        # Include spread data if available (for spread compression analysis)
        include_patterns = ['bid', 'ask', 'spread', 'compression']
        
        feature_columns = []
        for col in self.df.columns:
            # Skip target column
            if col == self.target_column:
                continue
            
            # Skip columns matching exclude patterns
            if any(pattern in col for pattern in exclude_patterns):
                continue
            
            # Skip exact matches
            if col in exclude_exact:
                continue
            
            # Include if it matches include patterns or is a numeric feature
            if (any(pattern in col for pattern in include_patterns) or 
                pd.api.types.is_numeric_dtype(self.df[col])):
                feature_columns.append(col)
        
        features_df = self.df[feature_columns].copy()
        
        # Handle missing values
        features_df = features_df.fillna(features_df.median())
        
        # Handle infinity values and extremely large values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Replace infinity values with NaN, then fill with median
            features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
            if features_df[col].isna().any():
                features_df[col] = features_df[col].fillna(features_df[col].median())
            
            # Cap extremely large values at 99.9th percentile
            upper_cap = features_df[col].quantile(0.999)
            lower_cap = features_df[col].quantile(0.001)
            features_df[col] = features_df[col].clip(lower_cap, upper_cap)
        
        # Remove constant features
        constant_features = [col for col in features_df.columns 
                           if features_df[col].nunique() <= 1]
        if constant_features:
            logger.info(f"Removing {len(constant_features)} constant features")
            features_df = features_df.drop(columns=constant_features)
        
        return features_df
    
    def _determine_task_type(self, task_type: str) -> str:
        """Determine if this is a regression or classification task"""
        
        if task_type != 'auto':
            return task_type
        
        # Check if target is binary/categorical
        unique_values = self.target.nunique()
        if unique_values <= 10 and self.target.dtype in ['bool', 'int64', 'int32']:
            return 'classification'
        else:
            return 'regression'
    
    def correlation_importance(self) -> pd.Series:
        """Feature importance based on correlation with target"""
        
        correlations = {}
        
        for feature in self.features.columns:
            try:
                # Use Spearman correlation for robust analysis
                corr, p_value = spearmanr(self.features[feature], self.target)
                correlations[feature] = abs(corr)  # Use absolute correlation
            except:
                correlations[feature] = 0.0
        
        result = pd.Series(correlations).sort_values(ascending=False)
        self.importance_results['correlation'] = result
        
        logger.info(f"Correlation analysis completed. Top feature: {result.index[0]} ({result.iloc[0]:.4f})")
        return result
    
    def mutual_information_importance(self) -> pd.Series:
        """Feature importance based on mutual information - OPTIMIZED for large datasets"""
        
        try:
            # Use sampling for large datasets to prevent system strain while preserving functionality
            features_sample = self.features
            target_sample = self.target
            
            if len(self.features) > 200000:
                logger.info(f"Large dataset detected ({len(self.features):,} rows). Using stratified sampling for MI analysis.")
                # Create combined dataframe for consistent sampling
                combined_df = pd.concat([self.features, self.target], axis=1)
                sampled_df = safe_sample_for_analysis(combined_df, max_rows=200000, preserve_distribution=True)
                
                features_sample = sampled_df.iloc[:, :-1]  # All columns except last
                target_sample = sampled_df.iloc[:, -1]     # Last column (target)
                
                logger.info(f"Using {len(features_sample):,} sampled rows for mutual information analysis")
            
            with PerformanceContext("Mutual Information Analysis"):
                if self.task_type == 'classification':
                    mi_scores = mutual_info_classif(features_sample, target_sample, random_state=42)
                else:
                    mi_scores = mutual_info_regression(features_sample, target_sample, random_state=42)
            
            result = pd.Series(mi_scores, index=self.features.columns).sort_values(ascending=False)
            self.importance_results['mutual_information'] = result
            
            logger.info(f"Mutual information analysis completed. Top feature: {result.index[0]} ({result.iloc[0]:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Mutual information analysis failed: {e}")
            return pd.Series(0, index=self.features.columns)
    
    def random_forest_importance(self) -> pd.Series:
        """Feature importance using Random Forest - OPTIMIZED for large datasets"""
        
        try:
            # Use sampling for large datasets to prevent memory issues
            features_sample = self.features
            target_sample = self.target
            
            if len(self.features) > 300000:
                logger.info(f"Large dataset detected ({len(self.features):,} rows). Using sampling for Random Forest analysis.")
                combined_df = pd.concat([self.features, self.target], axis=1)
                sampled_df = safe_sample_for_analysis(combined_df, max_rows=300000, preserve_distribution=True)
                
                features_sample = sampled_df.iloc[:, :-1]
                target_sample = sampled_df.iloc[:, -1]
                logger.info(f"Using {len(features_sample):,} sampled rows for Random Forest analysis")
            
            # Choose appropriate model with reduced complexity for very large datasets
            n_estimators = 50 if len(features_sample) > 100000 else 100
            
            if self.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            
            # Fit model with performance monitoring
            with PerformanceContext("Random Forest Training"):
                model.fit(features_sample, target_sample)
            
            # Get feature importances
            importances = model.feature_importances_
            result = pd.Series(importances, index=self.features.columns).sort_values(ascending=False)
            self.importance_results['random_forest'] = result
            
            logger.info(f"Random Forest analysis completed. Top feature: {result.index[0]} ({result.iloc[0]:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Random Forest analysis failed: {e}")
            return pd.Series(0, index=self.features.columns)
    
    def lasso_importance(self) -> pd.Series:
        """Feature importance using Lasso regularization (regression only)"""
        
        if self.task_type != 'regression':
            logger.info("Lasso importance skipped (classification task)")
            return pd.Series(0, index=self.features.columns)
        
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.features)
            
            # Fit Lasso with cross-validation
            lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
            lasso.fit(X_scaled, self.target)
            
            # Get absolute coefficients as importance
            importances = np.abs(lasso.coef_)
            result = pd.Series(importances, index=self.features.columns).sort_values(ascending=False)
            self.importance_results['lasso'] = result
            
            logger.info(f"Lasso analysis completed. Top feature: {result.index[0]} ({result.iloc[0]:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Lasso analysis failed: {e}")
            return pd.Series(0, index=self.features.columns)
    
    def permutation_importance_analysis(self) -> pd.Series:
        """Feature importance using permutation importance"""
        
        try:
            # Choose appropriate model
            if self.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            
            # Split data for unbiased permutation importance
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.target, test_size=0.3, random_state=42
            )
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
            )
            
            # Get mean importance scores
            importances = perm_importance.importances_mean
            result = pd.Series(importances, index=self.features.columns).sort_values(ascending=False)
            self.importance_results['permutation'] = result
            
            logger.info(f"Permutation importance analysis completed. Top feature: {result.index[0]} ({result.iloc[0]:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Permutation importance analysis failed: {e}")
            return pd.Series(0, index=self.features.columns)
    
    def analyze_all_methods(self) -> Dict[str, pd.Series]:
        """Run all importance analysis methods"""
        
        logger.info("Starting comprehensive feature importance analysis...")
        
        # Run all methods
        self.correlation_importance()
        self.mutual_information_importance()
        self.random_forest_importance()
        self.lasso_importance()
        self.permutation_importance_analysis()
        
        return self.importance_results
    
    def get_consensus_ranking(self, top_n: int = None) -> pd.Series:
        """Get consensus feature ranking across all methods"""
        
        if not self.importance_results:
            self.analyze_all_methods()
        
        # Normalize all importance scores to 0-1 range
        normalized_results = {}
        for method, scores in self.importance_results.items():
            if scores.sum() > 0:  # Only include methods that produced results
                normalized_results[method] = scores / scores.max()
        
        # Calculate mean ranking across methods
        if not normalized_results:
            logger.error("No valid importance results found")
            return pd.Series()
        
        # Combine all normalized scores
        combined_df = pd.DataFrame(normalized_results)
        consensus_scores = combined_df.mean(axis=1).sort_values(ascending=False)
        
        if top_n:
            consensus_scores = consensus_scores.head(top_n)
        
        logger.info(f"Consensus ranking completed. Top feature: {consensus_scores.index[0]} ({consensus_scores.iloc[0]:.4f})")
        return consensus_scores
    
    def plot_importance_comparison(self, top_n: int = 20, save_path: str = None) -> None:
        """Create comprehensive importance comparison plots"""
        
        if not self.importance_results:
            self.analyze_all_methods()
        
        # Get consensus ranking
        consensus = self.get_consensus_ranking(top_n)
        top_features = consensus.index.tolist()
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Feature Importance Analysis - {self.target_column} ({self.task_type})', 
                     fontsize=16, fontweight='bold')
        
        # Plot each method
        methods_to_plot = [(name, scores) for name, scores in self.importance_results.items() 
                          if scores.sum() > 0]
        
        for idx, (method, scores) in enumerate(methods_to_plot[:5]):
            row, col = idx // 3, idx % 3
            
            # Get top features for this method
            method_top = scores.head(top_n)
            
            axes[row, col].barh(range(len(method_top)), method_top.values)
            axes[row, col].set_yticks(range(len(method_top)))
            axes[row, col].set_yticklabels(method_top.index, fontsize=8)
            axes[row, col].set_title(f'{method.replace("_", " ").title()}')
            axes[row, col].invert_yaxis()
        
        # Plot consensus ranking
        axes[1, 2].barh(range(len(consensus)), consensus.values, color='orange')
        axes[1, 2].set_yticks(range(len(consensus)))
        axes[1, 2].set_yticklabels(consensus.index, fontsize=8)
        axes[1, 2].set_title('Consensus Ranking')
        axes[1, 2].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def create_summary_report(self, top_n: int = 15) -> str:
        """Create a detailed text summary of feature importance analysis"""
        
        if not self.importance_results:
            self.analyze_all_methods()
        
        consensus = self.get_consensus_ranking(top_n)
        
        report = []
        report.append("=" * 80)
        report.append("FEATURE IMPORTANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Dataset: {len(self.df)} rows, {len(self.features.columns)} features")
        report.append(f"Target: {self.target_column} ({self.task_type})")
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Consensus ranking
        report.append("CONSENSUS TOP FEATURES:")
        report.append("-" * 40)
        for i, (feature, score) in enumerate(consensus.head(top_n).items(), 1):
            report.append(f"{i:2d}. {feature:<30} {score:.4f}")
        report.append("")
        
        # Method-specific top 5
        for method, scores in self.importance_results.items():
            if scores.sum() > 0:
                report.append(f"{method.upper().replace('_', ' ')} - TOP 5:")
                report.append("-" * 30)
                for i, (feature, score) in enumerate(scores.head(5).items(), 1):
                    report.append(f"{i}. {feature:<25} {score:.4f}")
                report.append("")
        
        return "\n".join(report)
    
    def analyze_and_save(self, output_dir: str = "storage/data_engineering_stats/feature_importance") -> Dict:
        """Complete analysis with automatic saving"""
        
        # Create timestamped subdirectory
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = Path(output_dir) / timestamp
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run analysis
        results = self.analyze_all_methods()
        consensus = self.get_consensus_ranking()
        
        # Save plot
        plot_path = output_path / "feature_importance_analysis.png"
        self.plot_importance_comparison(save_path=str(plot_path))
        
        # Save detailed results
        results_path = output_path / "importance_results.csv"
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_path)
        
        # Save summary report
        report_path = output_path / "importance_report.txt"
        with open(report_path, 'w') as f:
            f.write(self.create_summary_report())
        
        # Save consensus ranking
        consensus_path = output_path / "consensus_ranking.csv"
        consensus.to_csv(consensus_path, header=['consensus_score'])
        
        logger.info(f"Feature importance analysis completed and saved to {output_path}")
        
        return {
            'consensus': consensus,
            'detailed_results': results,
            'output_directory': str(output_path),
            'files_created': [str(plot_path), str(results_path), str(report_path), str(consensus_path)]
        }


def analyze_processed_data(data_path: str, target_column: str = None, top_n: int = 20) -> Dict:
    """
    Convenience function to analyze processed data
    
    Args:
        data_path: Path to processed data file
        target_column: Target column name
        top_n: Number of top features to focus on
    
    Returns:
        Analysis results dictionary
    """
    analyzer = FeatureImportanceAnalyzer(data_path, target_column)
    return analyzer.analyze_and_save()


if __name__ == "__main__":
    # Example usage for spread compression hypothesis
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        target_col = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"Analyzing feature importance for: {data_path}")
        results = analyze_processed_data(data_path, target_col)
        print(f"Analysis completed. Results saved to: {results['output_directory']}")
    else:
        print("Usage: python feature_importance.py <data_path> [target_column]")