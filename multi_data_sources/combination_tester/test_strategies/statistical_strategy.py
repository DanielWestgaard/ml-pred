"""
StatisticalTestStrategy: Basic statistical analysis for combination testing

This strategy leverages existing data processing components for data quality
and focuses on correlation analysis, statistical significance testing,
and identifying patterns that indicate institutional blind spots.

Reuses: data/processing/validation.py and data/processing/cleaning.py
Keeps it simple but effective - the foundation for systematic inefficiency discovery.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy import stats
from utils.logging_utils import get_logger

# Reuse existing data processing components
from data.processing.validation import DataValidator
from data.processing.cleaning import DataCleaner

logger = get_logger()


class StatisticalTestStrategy:
    """
    Statistical testing strategy for data source combinations.
    
    Leverages existing data quality infrastructure and focuses on:
    - Data quality validation (reusing DataValidator)
    - Basic data cleaning (reusing DataCleaner) 
    - Correlation analysis between features
    - Statistical significance testing
    - Institutional blind spot scoring
    """
    
    def __init__(self, validation_config: Dict[str, Any] = None):
        """
        Initialize strategy with validation criteria.
        
        Args:
            validation_config: Dictionary containing validation thresholds
        """
        self.validation_config = validation_config or {}
        
        # Default validation thresholds (can be overridden in config)
        self.min_correlation = self.validation_config.get('min_correlation_threshold', 0.1)
        self.significance_level = self.validation_config.get('significance_level', 0.05)
        self.min_data_points = self.validation_config.get('min_data_points', 100)
        
        logger.debug(f"StatisticalTestStrategy initialized with thresholds: "
                    f"correlation={self.min_correlation}, significance={self.significance_level}")
    
    def test_combination(self, data: pd.DataFrame, combination: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a specific feature combination using statistical analysis.
        
        Args:
            data: DataFrame containing the features to test
            combination: Dictionary describing the combination being tested
            
        Returns:
            Dictionary containing test results and scores
        """
        combination_name = combination.get('name', 'unknown')
        features = combination.get('features', [])
        
        logger.debug(f"Testing combination: {combination_name} with features: {features}")
        
        # Step 1: Data quality validation using existing components
        quality_check = self._validate_data_quality(data)
        if not quality_check['is_valid']:
            return {
                'valid': False,
                'error': f"Data quality validation failed: {quality_check['summary']}",
                'composite_score': 0.0,
                'validation_issues': quality_check['issues']
            }
        
        # Step 2: Light data cleaning if needed
        clean_data = self._ensure_data_quality(data)
        
        # Step 3: Basic validation checks
        if len(clean_data) < self.min_data_points:
            return {
                'valid': False,
                'error': f'Insufficient data points: {len(clean_data)} < {self.min_data_points}',
                'composite_score': 0.0
            }
        
        try:
            # Core statistical analysis
            correlation_results = self._analyze_correlations(clean_data)
            significance_results = self._test_significance(clean_data)
            pattern_results = self._analyze_patterns(clean_data)
            blind_spot_score = self._calculate_blind_spot_score(features)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                correlation_results, significance_results, pattern_results, blind_spot_score
            )
            
            # Validation check
            is_valid = self._validate_combination(
                correlation_results, significance_results, len(clean_data)
            )
            
            return {
                'valid': is_valid,
                'data_points': len(clean_data),
                'data_quality': quality_check,
                'correlation_analysis': correlation_results,
                'significance_analysis': significance_results,
                'pattern_analysis': pattern_results,
                'blind_spot_score': blind_spot_score,
                'composite_score': composite_score,
                'max_correlation': correlation_results.get('max_correlation', 0),
                'min_p_value': significance_results.get('min_p_value', 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error in statistical testing for {combination_name}: {e}")
            return {
                'valid': False,
                'error': str(e),
                'composite_score': 0.0
            }
    
    def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality using existing DataValidator component.
        
        Returns summary instead of full validation to keep it lightweight.
        """
        try:
            # Use existing DataValidator for thorough validation
            validator = DataValidator(data)
            validation_result = validator.run()
            
            # Create summary for combination testing context
            issues = validation_result.get('issues', [])
            critical_issues = [issue for issue in issues 
                             if issue.get('type') in ['Invalid Prices', 'Infinite Values', 'Missing Required Columns']]
            
            return {
                'is_valid': len(critical_issues) == 0,  # Only fail on critical issues
                'total_issues': len(issues),
                'critical_issues': len(critical_issues),
                'summary': f"{len(critical_issues)} critical issues, {len(issues)} total issues",
                'issues': issues[:3]  # Keep first 3 issues for context
            }
            
        except Exception as e:
            logger.warning(f"Data quality validation failed: {e}")
            return {
                'is_valid': True,  # Don't fail combination testing due to validation errors
                'error': str(e),
                'summary': 'Validation check failed, proceeding with analysis'
            }
    
    def _ensure_data_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply light data cleaning using existing DataCleaner if needed.
        
        Only applies essential fixes to avoid changing the data too much.
        """
        try:
            # Check if data needs basic cleaning
            needs_cleaning = (
                data.isnull().any().any() or  # Has missing values
                np.isinf(data.select_dtypes(include=[np.number]).values).any() or  # Has infinite values
                data.index.duplicated().any()  # Has duplicate indices
            )
            
            if not needs_cleaning:
                return data.copy()
            
            logger.debug("Applying light data cleaning for combination testing")
            
            # Create a copy to avoid modifying original data
            clean_data = data.copy()
            
            # Apply only essential cleaning steps
            # Handle missing values (simple forward fill)
            clean_data = clean_data.ffill().bfill()
            
            # Handle infinite values (replace with NaN then fill)
            numeric_cols = clean_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                clean_data[col] = clean_data[col].replace([np.inf, -np.inf], np.nan)
                clean_data[col] = clean_data[col].ffill().bfill()
                # Last resort: use median
                if clean_data[col].isnull().any():
                    clean_data[col] = clean_data[col].fillna(clean_data[col].median())
            
            # Remove duplicate indices
            if clean_data.index.duplicated().any():
                clean_data = clean_data[~clean_data.index.duplicated(keep='first')]
            
            return clean_data
            
        except Exception as e:
            logger.warning(f"Light data cleaning failed: {e}, using original data")
            return data.copy()
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between features."""
        try:
            # Focus on numeric columns only
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) < 2:
                return {'max_correlation': 0, 'average_correlation': 0, 'error': 'Insufficient numeric columns'}
            
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()
            
            # Find maximum absolute correlation (excluding diagonal)
            np.fill_diagonal(corr_matrix.values, 0)
            max_corr = np.abs(corr_matrix.values).max()
            
            # Find average correlation strength
            upper_triangle = np.triu(np.abs(corr_matrix.values), k=1)
            avg_corr = np.mean(upper_triangle[upper_triangle > 0]) if np.any(upper_triangle > 0) else 0
            
            return {
                'max_correlation': round(max_corr, 4),
                'average_correlation': round(avg_corr, 4),
                'correlation_matrix_size': corr_matrix.shape[0],
                'strong_correlations': int(np.sum(np.abs(corr_matrix.values) > 0.3)) // 2  # Divide by 2 for symmetry
            }
            
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
            return {'max_correlation': 0, 'average_correlation': 0, 'error': str(e)}
    
    def _test_significance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test statistical significance of relationships."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) < 2:
                return {'min_p_value': 1.0, 'total_significant': 0, 'error': 'Insufficient numeric columns'}
            
            # Assume first numeric column is target (typically Close price)
            target_col = numeric_data.columns[0]
            target = numeric_data[target_col]
            
            p_values = []
            significant_features = []
            
            for col in numeric_data.columns[1:]:  # Skip target column
                try:
                    # Correlation test
                    corr, p_val = stats.pearsonr(target, numeric_data[col])
                    p_values.append(p_val)
                    
                    if p_val < self.significance_level:
                        significant_features.append(col)
                        
                except Exception:
                    continue  # Skip problematic columns
            
            min_p_value = min(p_values) if p_values else 1.0
            
            return {
                'min_p_value': round(min_p_value, 6),
                'significant_features': significant_features,
                'total_significant': len(significant_features),
                'significance_ratio': round(len(significant_features) / max(len(numeric_data.columns) - 1, 1), 3)
            }
            
        except Exception as e:
            logger.warning(f"Significance testing failed: {e}")
            return {'min_p_value': 1.0, 'total_significant': 0, 'error': str(e)}
    
    def _analyze_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze pattern consistency and stability."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            # Simple pattern analysis: look for consistent relationships over time
            if len(numeric_data) < 100 or len(numeric_data.columns) < 2:
                return {'pattern_consistency': 0, 'stability_score': 0}
            
            quarter_size = len(numeric_data) // 4
            correlations_over_time = []
            
            for i in range(4):
                start_idx = i * quarter_size
                end_idx = start_idx + quarter_size if i < 3 else len(numeric_data)
                quarter_data = numeric_data.iloc[start_idx:end_idx]
                
                if len(quarter_data) > 10:  # Minimum data for correlation
                    try:
                        # Calculate max correlation for this quarter
                        corr_matrix = quarter_data.corr()
                        np.fill_diagonal(corr_matrix.values, 0)
                        max_corr = np.abs(corr_matrix.values).max()
                        correlations_over_time.append(max_corr)
                    except Exception:
                        continue
            
            if correlations_over_time:
                # Pattern consistency: how stable are correlations over time
                consistency = 1 - (np.std(correlations_over_time) / (np.mean(correlations_over_time) + 0.01))
                consistency = max(0, min(1, consistency))  # Bound between 0 and 1
                
                return {
                    'pattern_consistency': round(consistency, 4),
                    'stability_score': round(np.mean(correlations_over_time), 4),
                    'quarters_analyzed': len(correlations_over_time)
                }
            else:
                return {'pattern_consistency': 0, 'stability_score': 0}
                
        except Exception as e:
            logger.warning(f"Pattern analysis failed: {e}")
            return {'pattern_consistency': 0, 'stability_score': 0}
    
    def _calculate_blind_spot_score(self, features: List[str]) -> Dict[str, Any]:
        """
        Calculate institutional blind spot score.
        
        Higher scores indicate combinations institutions are likely to miss
        due to organizational constraints.
        """
        try:
            # Score based on complexity and cross-data-type combination
            complexity_score = min(len(features) / 5.0, 1.0)  # More features = higher complexity
            
            # Cross-data-type score: bonus for combining different data types
            data_types = set()
            for feature in features:
                feature_lower = feature.lower()
                if any(market_term in feature_lower for market_term in ['close', 'volume', 'spread', 'price', 'open', 'high', 'low']):
                    data_types.add('market')
                elif any(calendar_term in feature_lower for calendar_term in ['earnings', 'options', 'holiday', 'vacation', 'quarter']):
                    data_types.add('calendar')
                elif any(sentiment_term in feature_lower for sentiment_term in ['sentiment', 'news', 'social']):
                    data_types.add('sentiment')
            
            cross_type_score = min(len(data_types) / 3.0, 1.0)  # Max score for 3+ data types
            
            # Organizational silo penalty: institutions struggle with cross-silo combinations
            silo_penalty = 0.2 if len(data_types) > 1 else 0
            
            overall_score = (complexity_score * 0.4 + cross_type_score * 0.4 + silo_penalty * 0.2)
            
            return {
                'overall_score': round(overall_score, 4),
                'complexity_score': round(complexity_score, 4),
                'cross_type_score': round(cross_type_score, 4),
                'data_types_identified': list(data_types),
                'institutional_challenge': len(data_types) > 1
            }
            
        except Exception as e:
            logger.warning(f"Blind spot scoring failed: {e}")
            return {'overall_score': 0}
    
    def _calculate_composite_score(self, correlation_results: Dict, significance_results: Dict, 
                                 pattern_results: Dict, blind_spot_score: Dict) -> float:
        """Calculate overall composite score for the combination."""
        try:
            # Extract key metrics
            max_corr = correlation_results.get('max_correlation', 0)
            min_p_val = significance_results.get('min_p_value', 1.0)
            pattern_consistency = pattern_results.get('pattern_consistency', 0)
            blind_spot = blind_spot_score.get('overall_score', 0)
            
            # Convert p-value to score (lower p-value = higher score)
            significance_score = max(0, 1 - (min_p_val / self.significance_level))
            
            # Weighted composite score
            composite = (
                max_corr * 0.3 +           # Statistical strength
                significance_score * 0.3 +  # Statistical significance  
                pattern_consistency * 0.2 + # Pattern stability
                blind_spot * 0.2           # Institutional blind spot potential
            )
            
            return round(composite, 4)
            
        except Exception as e:
            logger.warning(f"Composite score calculation failed: {e}")
            return 0.0
    
    def _validate_combination(self, correlation_results: Dict, significance_results: Dict, 
                            data_points: int) -> bool:
        """Validate if combination meets minimum criteria."""
        try:
            # Check all validation criteria
            has_min_correlation = correlation_results.get('max_correlation', 0) >= self.min_correlation
            has_significance = significance_results.get('min_p_value', 1.0) <= self.significance_level
            has_min_data = data_points >= self.min_data_points
            
            return has_min_correlation and has_significance and has_min_data
            
        except Exception:
            return False