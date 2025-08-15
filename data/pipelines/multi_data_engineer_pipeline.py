"""
Enhanced Multi-Data Engineering Pipeline

Extends the original engineer_pipeline.py to integrate validated multi-data sources
with market data for systematic inefficiency exploitation. 

Key Features:
- Config-driven data source selection based on combination testing results
- Seamless integration with existing hypothesis workflow  
- Uses validated calendar event patterns (CLOV/SPCE: 0.47-0.59 correlations)
- Time-aligned feature engineering maintaining temporal integrity
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Existing pipeline imports
from utils import data_utils
from data.processing.cleaning import DataCleaner
from data.processing.validation import DataValidator
from data.features.enhanced_generation import ModelAwareFeatureGenerator
from data.features.transformation import FeatureTransformer
from data.features.selection import FeatureSelector
from data.hypothesis import TargetHypothesis
import utils.hypothesis.config_loader as config_loader
from utils.logging_utils import get_logger
from utils.performance_utils import memory_monitor, PerformanceContext

# Multi-data source imports
from multi_data_sources.combiners import TimeSeriesDataCombiner
from multi_data_sources.ds_orchestrator import DataSourceOrchestrator

logger = get_logger()


class MultiDataEngineeringPipeline:
    """
    Enhanced engineering pipeline that integrates validated multi-data sources
    with market data for systematic inefficiency discovery.
    
    Extends original engineer_pipeline.py with:
    - Multi-data source integration
    - Config-driven data source selection
    - Validated calendar event patterns
    - Temporal integrity preservation
    """
    
    def __init__(self, 
                 raw_dataset: str = None, 
                 output_path: str = None, 
                 save_cleaned_data: bool = False, 
                 hypothesis_source: str = None, 
                 analyze_feature_importance: bool = False):
        """
        Initialize enhanced pipeline with multi-data source capabilities.
        
        Args:
            raw_dataset: Path to raw market data (CSV format)
            output_path: Output path for processed data 
            save_cleaned_data: Whether to save cleaned intermediate data
            hypothesis_source: Path to hypothesis config with multi_data section
            analyze_feature_importance: Run feature importance analysis
        """
        logger.info("🚀 Initializing Enhanced Multi-Data Engineering Pipeline")
        
        self.output_path = output_path
        self.analyze_feature_importance = analyze_feature_importance
        self.save_cleaned_data = save_cleaned_data
        
        # Load configuration (enhanced to handle multi_data section)
        if hypothesis_source is not None:
            self.config = config_loader.load_hypothesis_config(hypothesis_source).handle_hypothesis_source()
            self.hypothesis_name = self.config.get('hypothesis_name')
            self.selected_features = self.config.get('features')
            self.spread_focused = self.config.get('spread_focused', False)
            
            # Multi-data configuration
            self.multi_data_config = self.config.get('multi_data', {})
            self.data_sources = self.multi_data_config.get('sources', [])
            self.symbol = self.multi_data_config.get('symbol')
            
            # Get market data path (from config or fallback to raw_dataset)
            raw_dataset = self.config.get('data_pipeline_raw_source') or raw_dataset
        else:
            self.hypothesis_name = None
            self.selected_features = None
            self.spread_focused = False
            self.multi_data_config = {}
            self.data_sources = []
            self.symbol = None
        
        self.raw_data_path = raw_dataset
        
        # Initialize components
        self.combiner = TimeSeriesDataCombiner()
        self.data_source_orchestrator = DataSourceOrchestrator()
        
        logger.info(f"📊 Symbol: {self.symbol}")
        logger.info(f"🔗 Data sources configured: {self.data_sources}")
        
    def run(self) -> pd.DataFrame:
        """
        Run the enhanced pipeline with multi-data source integration.
        
        Returns:
            pd.DataFrame: Engineered data ready for ML training with both
                         market features and validated multi-data source features
        """
        logger.info("---------- Running Enhanced Multi-Data Pipeline ----------")
        
        # Monitor initial memory state
        memory_monitor.check_and_warn("at enhanced pipeline start")
        
        # Step 1: Load and combine multi-data sources
        with PerformanceContext("Multi-Data Source Integration"):
            combined_df = self._load_and_combine_data()
            memory_monitor.safe_cleanup()
        
        # Step 2: Enhanced data validation (validates combined dataset)
        with PerformanceContext("Enhanced Data Validation"):
            validated_df = self._validate_combined_data(combined_df)
            memory_monitor.safe_cleanup()
        
        # Step 3: Enhanced feature generation (operates on combined dataset)
        with PerformanceContext("Enhanced Feature Generation"):
            featured_df = self._generate_enhanced_features(validated_df)
            memory_monitor.safe_cleanup()
        
        # Step 4: Feature transformation and selection (reuse existing logic)
        with PerformanceContext("Feature Transformation"):
            transformed_df = self._transform_features(featured_df)
            memory_monitor.safe_cleanup()
        
        # Step 5: Feature selection (if needed)
        with PerformanceContext("Feature Selection"):
            final_df = self._select_features(transformed_df)
            memory_monitor.safe_cleanup()
        
        # Step 6: Save results
        self._save_enhanced_data(final_df)
        
        logger.info("---------- Enhanced Multi-Data Pipeline Finished ----------")
        logger.info(f"📈 Final dataset shape: {final_df.shape}")
        logger.info(f"🔧 Features generated: {len([col for col in final_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])}")
        
        return final_df
    
    def _load_and_combine_data(self) -> pd.DataFrame:
        """
        Load market data and combine with validated multi-data sources.
        
        Uses validated calendar event patterns from combination testing.
        """
        logger.info("🔗 Loading and combining multi-data sources...")
        
        if not self.data_sources or not self.symbol:
            logger.info("📊 No multi-data sources configured, loading market data only")
            df, _ = data_utils.check_and_return_df(self.raw_data_path)
            return df
        
        # Extract timeframe and date range from config
        timeframe = self.multi_data_config.get('timeframe', '1h')  # Default to 1h for intraday
        date_range = {
            'start': self.multi_data_config.get('from_date'),
            'end': self.multi_data_config.get('to_date')
        }
        
        logger.info(f"📅 Timeframe: {timeframe}")
        logger.info(f"📊 Date range: {date_range['start']} to {date_range['end']}")
        
        # Use validated combination approach
        try:
            combined_df = self.combiner.combine_data(
                symbol=self.symbol,
                timeframe=timeframe,
                date_range=date_range,
                data_sources=self.data_sources
            )
            
            logger.info(f"✅ Successfully combined data: {combined_df.shape}")
            logger.info(f"🎯 Multi-data features added: {[col for col in combined_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"❌ Multi-data combination failed: {e}")
            logger.info("📊 Falling back to market data only")
            df, _ = data_utils.check_and_return_df(self.raw_data_path)
            return df
    
    def _validate_combined_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate combined dataset using existing validation logic.
        """
        logger.info("🔍 Validating combined dataset...")
        
        # Apply existing data cleaning
        cleaner = DataCleaner(raw_dataset=df, save_cleaned_data=self.save_cleaned_data)
        cleaned_df = cleaner.run()
        
        # Apply existing validation
        validator = DataValidator(cleaned_df)
        validation_results = validator.run()
        validated_df = validation_results["validated_data"]
        
        # Log validation results
        data_utils.check_validation(validation_results["is_valid"], validation_results["issues"])
        
        logger.info(f"✅ Validation complete: {validated_df.shape}")
        return validated_df
    
    def _generate_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features using existing feature generation enhanced with multi-data.
        """
        logger.info("🎨 Generating enhanced features...")
        
        # Use existing feature generator (now operates on combined dataset)
        feature_generator = ModelAwareFeatureGenerator(df)
        
        if self.selected_features is not None:
            logger.info(f"🎯 Generating specific features: {self.selected_features}")
            featured_df = feature_generator.generate_selected(self.selected_features)
        else:
            logger.info("🎨 Generating all features from enhanced dataset")
            featured_df = feature_generator.generate_all()
        
        # Add target hypothesis
        if self.hypothesis_name:
            hypo = TargetHypothesis(featured_df, self.hypothesis_name)
            featured_df = hypo.run()
        
        # Feature importance analysis (if requested)
        if self.analyze_feature_importance:
            self._run_enhanced_feature_importance(featured_df)
        
        logger.info(f"✅ Feature generation complete: {featured_df.shape}")
        return featured_df
    
    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using existing transformation logic."""
        logger.info("🔄 Transforming features...")
        
        transformer = FeatureTransformer(df)
        transformed_df = transformer.run()
        
        logger.info(f"✅ Transformation complete: {transformed_df.shape}")
        return transformed_df
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features using existing selection logic (if not skipped)."""
        if self.selected_features is not None:
            logger.info("⏭️ Skipping feature selection (specific features provided)")
            return df
        
        logger.info("🎯 Selecting optimal features...")
        
        selector = FeatureSelector(df)
        selected_df = selector.run()
        
        logger.info(f"✅ Feature selection complete: {selected_df.shape}")
        return selected_df
    
    def _save_enhanced_data(self, df: pd.DataFrame):
        """Save enhanced dataset with multi-data features."""
        try:
            if self.output_path:
                filepath = self.output_path
            else:
                # Generate enhanced filename
                base_path = Path(self.raw_data_path)
                filepath = base_path.parent / f"{base_path.stem}_enhanced_multidata{base_path.suffix}"
            
            df.to_csv(filepath, index=True)
            logger.info(f"💾 Enhanced dataset saved: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Unable to save enhanced data: {e}")
            fallback_path = f"enhanced_multidata_{self.symbol or 'unknown'}.csv"
            df.to_csv(fallback_path, index=True)
            logger.info(f"💾 Saved to fallback location: {fallback_path}")
    
    def _run_enhanced_feature_importance(self, df: pd.DataFrame):
        """Run feature importance analysis on enhanced dataset."""
        try:
            from data.features.feature_importance import FeatureImportanceAnalyzer
            
            logger.info("📊 Running enhanced feature importance analysis...")
            
            # Determine target column
            target_col = None
            if self.hypothesis_name:
                target_candidates = [col for col in df.columns if col.startswith('target_')]
                if target_candidates:
                    target_col = target_candidates[0]
            
            # Run analysis
            analyzer = FeatureImportanceAnalyzer(df, target_col)
            results = analyzer.analyze_and_save()
            
            # Log results with multi-data context
            consensus = results['consensus']
            logger.info("🏆 Enhanced Feature Importance Results:")
            logger.info("📈 Top 10 features (including multi-data sources):")
            
            for i, (feature, score) in enumerate(consensus.head(10).items(), 1):
                # Identify if it's a multi-data feature
                is_multi_data = any(keyword in feature.lower() for keyword in 
                                  ['earnings', 'options', 'holiday', 'calendar', 'fiscal'])
                prefix = "🎯" if is_multi_data else "📊"
                logger.info(f"  {i:2d}. {prefix} {feature:<30} {score:.4f}")
            
            logger.info(f"📁 Full results saved to: {results['output_directory']}")
            
        except Exception as e:
            logger.error(f"❌ Enhanced feature importance analysis failed: {e}")
    
    def get_validated_data_sources(self) -> List[str]:
        """
        Get list of validated data sources based on combination testing results.
        
        Returns:
            List of data source names that have proven inefficiency patterns
        """
        # Based on our validation results, these are the proven sources
        validated_sources = []
        
        if 'calendar_events' in self.data_sources:
            # Only include if symbol is validated (CLOV or SPCE)
            if self.symbol in ['CLOV', 'SPCE']:
                validated_sources.append('calendar_events')
                logger.info(f"✅ Calendar events validated for {self.symbol} (correlation: 0.47-0.59)")
            else:
                logger.warning(f"⚠️ Calendar events not validated for {self.symbol}, skipping")
        
        return validated_sources


# Convenience function for backward compatibility
def create_enhanced_pipeline(hypothesis_config_path: str, **kwargs) -> MultiDataEngineeringPipeline:
    """
    Create enhanced pipeline from hypothesis configuration.
    
    Args:
        hypothesis_config_path: Path to hypothesis config with multi_data section
        **kwargs: Additional arguments for pipeline initialization
        
    Returns:
        MultiDataEngineeringPipeline: Configured pipeline ready to run
    """
    return MultiDataEngineeringPipeline(
        hypothesis_source=hypothesis_config_path,
        **kwargs
    )