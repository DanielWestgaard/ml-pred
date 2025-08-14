"""
CombinationTestOrchestrator: Main coordinator for systematic combination testing

This orchestrator manages the testing of different data source combinations
to identify market inefficiencies. It integrates with the existing hypothesis-driven
workflow while maintaining modularity and extensibility.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

from utils.hypothesis.config_loader import LoadHypothesisConfigMetadata
from utils.logging_utils import get_logger
from ..combiners import TimeSeriesDataCombiner
from .test_strategies import StatisticalTestStrategy

logger = get_logger()


class CombinationTestOrchestrator:
    """
    Orchestrates systematic testing of data source combinations.
    
    Simple, modular design following the project's architectural patterns.
    Focuses on finding institutional blind spots through combination analysis.
    """
    
    def __init__(self, hypothesis_config_path: str):
        """
        Initialize orchestrator with hypothesis configuration.
        
        Args:
            hypothesis_config_path: Path to hypothesis configuration file
        """
        self.config_path = hypothesis_config_path
        self.config_loader = LoadHypothesisConfigMetadata(hypothesis_config_path)
        self.config = self.config_loader.handle_hypothesis_source()
        
        # Extract combination testing config
        self.combination_config = self.config.get('combination_testing', {})
        if not self.combination_config:
            raise ValueError("No 'combination_testing' section found in configuration")
        
        # Initialize components
        self.combiner = TimeSeriesDataCombiner()
        self.test_strategy = self._initialize_test_strategy()
        
        # Results storage
        self.results_directory = self._setup_results_directory()
        
        logger.info(f"CombinationTestOrchestrator initialized for {self.config.get('hypothesis_name', 'unknown')}")
    
    def _initialize_test_strategy(self):
        """Initialize the appropriate test strategy based on configuration."""
        strategy_name = self.combination_config.get('strategy', 'statistical')
        
        if strategy_name == 'statistical':
            return StatisticalTestStrategy(self.combination_config.get('validation', {}))
        else:
            # Future: MLTestStrategy, CustomTestStrategy
            logger.warning(f"Unknown strategy '{strategy_name}', falling back to statistical")
            return StatisticalTestStrategy(self.combination_config.get('validation', {}))
    
    def _setup_results_directory(self) -> Path:
        """Setup results directory following the project's storage patterns."""
        symbol = self.config.get('multi_data_symbol', 'UNKNOWN')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy = self.combination_config.get('strategy', 'statistical')
        
        results_dir = Path(f"storage/combination_testing/{symbol}/{timestamp}_{strategy}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (results_dir / "results").mkdir(exist_ok=True)
        (results_dir / "visualizations").mkdir(exist_ok=True)
        (results_dir / "raw_test_data").mkdir(exist_ok=True)
        
        logger.info(f"Results directory created: {results_dir}")
        return results_dir
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete combination testing process.
        
        Returns:
            Dict containing comprehensive results and analysis
        """
        logger.info("🚀 Starting systematic combination testing...")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load combined data from multi-data sources
            combined_data = self._load_combined_data()
            logger.info(f"Loaded combined dataset: {len(combined_data)} rows, {len(combined_data.columns)} features")
            
            # Step 2: Generate test combinations
            test_combinations = self._generate_test_combinations(combined_data)
            logger.info(f"Generated {len(test_combinations)} test combinations")
            
            # Step 3: Test each combination
            test_results = self._test_combinations(combined_data, test_combinations)
            
            # Step 4: Analyze and rank results
            analysis_results = self._analyze_results(test_results)
            
            # Step 5: Save results
            comprehensive_results = {
                'summary': self._create_summary(test_results, start_time),
                'test_results': test_results,
                'analysis': analysis_results,
                'config_used': self.combination_config,
                'symbol': self.config.get('multi_data_symbol'),
                'timestamp': datetime.now().isoformat()
            }
            
            self._save_results(comprehensive_results)
            
            logger.info("✅ Combination testing completed successfully")
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"❌ Combination testing failed: {e}")
            raise
    
    def _load_combined_data(self) -> pd.DataFrame:
        """Load combined data using existing combiner infrastructure."""
        # Extract multi-data configuration
        symbol = self.config.get('multi_data_symbol')
        timeframe = '1d'  # Default to daily for combination testing
        date_range = {
            'start': self.config.get('multi_data_from_date'),
            'end': self.config.get('multi_data_to_date')
        }
        data_sources = self.config.get('multi_data_sources', [])
        
        if not symbol:
            raise ValueError("Missing required multi_data configuration (symbol)")
        
        logger.info(f"Loading combined data for {symbol} with sources: {data_sources}")
        
        # Use existing combiner to load data
        if data_sources:
            # Has additional data sources to combine
            combined_df = self.combiner.combine_data(
                symbol=symbol,
                timeframe=timeframe,
                date_range=date_range,
                data_sources=data_sources
            )
        else:
            # Just load market data for basic market-only combination testing
            logger.info("No additional data sources specified, loading market data only")
            combined_df = self.combiner._load_market_data(symbol, timeframe, date_range)
        
        return combined_df
    
    def _generate_test_combinations(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate combinations to test based on configuration."""
        test_matrix = self.combination_config.get('test_matrix', {})
        base_features = test_matrix.get('base_features', ['Close', 'Volume'])
        
        # Get available calendar features from the data
        calendar_features = [col for col in data.columns 
                           if any(keyword in col.lower() for keyword in 
                                 ['earnings', 'options', 'holiday', 'vacation', 'quarter'])]
        
        if not calendar_features:
            logger.warning("No calendar features found in combined data")
            return []
        
        logger.info(f"Available calendar features: {calendar_features}")
        
        # Generate combinations: base features + different calendar feature combinations
        combinations = []
        
        # Base only
        combinations.append({
            'name': 'market_only',
            'features': base_features,
            'description': 'Market data only (baseline)'
        })
        
        # Single calendar features
        for feature in calendar_features[:5]:  # Limit to first 5 to keep it simple
            combinations.append({
                'name': f'market_plus_{feature.lower().replace(" ", "_")}',
                'features': base_features + [feature],
                'description': f'Market data + {feature}'
            })
        
        # Multiple calendar features (2-3 combinations)
        if len(calendar_features) >= 2:
            combinations.append({
                'name': 'market_plus_multiple_calendar',
                'features': base_features + calendar_features[:3],  # First 3 calendar features
                'description': 'Market data + multiple calendar events'
            })
        
        logger.info(f"Generated {len(combinations)} test combinations")
        return combinations
    
    def _test_combinations(self, data: pd.DataFrame, combinations: List[Dict]) -> List[Dict]:
        """Test each combination using the configured strategy."""
        results = []
        
        for i, combination in enumerate(combinations, 1):
            logger.info(f"Testing combination {i}/{len(combinations)}: {combination['name']}")
            
            try:
                # Extract features for this combination
                features = combination['features']
                available_features = [f for f in features if f in data.columns]
                
                if len(available_features) < len(features):
                    missing = set(features) - set(available_features)
                    logger.warning(f"Missing features for {combination['name']}: {missing}")
                
                if len(available_features) < 2:
                    logger.warning(f"Insufficient features for {combination['name']}, skipping")
                    continue
                
                # Test this combination
                combination_data = data[available_features].copy()
                test_result = self.test_strategy.test_combination(combination_data, combination)
                
                # Save raw test data
                self._save_combination_data(combination_data, combination['name'])
                
                results.append({
                    'combination': combination,
                    'features_tested': available_features,
                    'test_result': test_result,
                    'valid': test_result.get('valid', False)
                })
                
            except Exception as e:
                logger.error(f"Error testing combination {combination['name']}: {e}")
                results.append({
                    'combination': combination,
                    'features_tested': [],
                    'test_result': {'error': str(e), 'valid': False},
                    'valid': False
                })
        
        return results
    
    def _analyze_results(self, test_results: List[Dict]) -> Dict[str, Any]:
        """Analyze and rank test results."""
        valid_results = [r for r in test_results if r.get('valid', False)]
        
        if not valid_results:
            return {'error': 'No valid test results to analyze'}
        
        # Sort by composite score (descending)
        scored_results = []
        for result in valid_results:
            score = result['test_result'].get('composite_score', 0)
            scored_results.append({
                'combination_name': result['combination']['name'],
                'features': result['features_tested'],
                'score': score,
                'correlation': result['test_result'].get('max_correlation', 0),
                'significance': result['test_result'].get('min_p_value', 1.0),
                'description': result['combination']['description']
            })
        
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'total_combinations': len(test_results),
            'valid_combinations': len(valid_results),
            'ranked_combinations': scored_results,
            'top_combinations': scored_results[:5],  # Top 5
            'institutional_blind_spots': len([r for r in scored_results if len(r['features']) > 2])
        }
    
    def _create_summary(self, test_results: List[Dict], start_time: datetime) -> Dict[str, Any]:
        """Create execution summary."""
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        valid_results = [r for r in test_results if r.get('valid', False)]
        top_score = max([r['test_result'].get('composite_score', 0) for r in valid_results], default=0)
        
        return {
            'total_combinations': len(test_results),
            'valid_combinations': len(valid_results),
            'execution_time_seconds': round(execution_time, 2),
            'top_score': round(top_score, 4),
            'symbol': self.config.get('multi_data_symbol'),
            'strategy_used': self.combination_config.get('strategy', 'statistical')
        }
    
    def _save_combination_data(self, data: pd.DataFrame, combination_name: str):
        """Save raw combination data for further analysis."""
        filepath = self.results_directory / "raw_test_data" / f"{combination_name}.csv"
        data.to_csv(filepath)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save comprehensive results to storage."""
        # Save main results file
        results_file = self.results_directory / "results" / "combination_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save ranked combinations as CSV for easy viewing
        if 'analysis' in results and 'ranked_combinations' in results['analysis']:
            ranked_df = pd.DataFrame(results['analysis']['ranked_combinations'])
            ranked_file = self.results_directory / "results" / "ranked_combinations.csv"
            ranked_df.to_csv(ranked_file, index=False)
        
        # Save configuration used
        config_file = self.results_directory / "config_used.yml"
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump({'combination_testing': self.combination_config}, f)
        
        logger.info(f"Results saved to: {self.results_directory}")
    
    def get_top_combinations(self, limit: int = 5) -> List[Dict]:
        """Get top combinations from last run (utility method)."""
        try:
            results_file = self.results_directory / "results" / "combination_test_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                return results.get('analysis', {}).get('top_combinations', [])[:limit]
        except Exception as e:
            logger.error(f"Error loading top combinations: {e}")
        
        return []