"""
Integration Adapter for Enhanced Feature Generation
===================================================

This module provides a seamless way to integrate the new enhanced
feature generation system with the existing codebase without breaking
current functionality.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd

from .enhanced_generation import ModelAwareFeatureGenerator


class EnhancedFeatureGeneratorAdapter:
    """
    Adapter class that provides backward compatibility while using
    the enhanced feature generation system under the hood
    """
    
    def __init__(self, dataset, selected_features: Union[Dict, List, None] = None):
        """
        Initialize adapter with backward compatibility
        
        Args:
            dataset: Input dataset
            selected_features: Can be Dict (old format) or List (new format) or None
        """
        self.dataset = dataset
        
        # Handle different selected_features formats
        if isinstance(selected_features, dict):
            # Old format: dict with feature names as keys
            self.selected_features = list(selected_features.keys())
            logging.info("Converted old dict format to new list format")
        elif isinstance(selected_features, list):
            # New format: list of feature names
            self.selected_features = selected_features
        else:
            # None - generate all features
            self.selected_features = None
        
        # Initialize the enhanced generator
        self.enhanced_generator = ModelAwareFeatureGenerator(
            dataset=dataset,
            selected_features=self.selected_features
        )
    
    def run(self) -> pd.DataFrame:
        """
        Run feature generation with enhanced system
        
        Returns:
            DataFrame with generated features
        """
        logging.info("Using Enhanced Feature Generation System")
        
        if self.selected_features:
            return self.enhanced_generator.generate_selected(self.selected_features)
        else:
            return self.enhanced_generator.generate_all()
    
    def generate_for_model(self, model_metadata_path: str) -> pd.DataFrame:
        """
        Generate features automatically based on model metadata
        
        Args:
            model_metadata_path: Path to model metadata file
            
        Returns:
            DataFrame with generated features
        """
        return self.enhanced_generator.generate_for_model(model_metadata_path)
    
    @property
    def df(self) -> pd.DataFrame:
        """Access to the dataframe for backward compatibility"""
        return self.enhanced_generator.df
    
    @classmethod
    def get_available_features(cls) -> List[str]:
        """Get all available features"""
        return ModelAwareFeatureGenerator.get_available_features()
    
    @classmethod
    def get_available_generators(cls) -> List[str]:
        """Get all available generators"""
        return ModelAwareFeatureGenerator.get_available_generators()


def create_enhanced_generator(dataset, selected_features=None, model_metadata_path=None):
    """
    Factory function to create the appropriate feature generator
    
    Args:
        dataset: Input dataset
        selected_features: Features to generate (dict or list format)
        model_metadata_path: Optional model metadata for automatic generation
        
    Returns:
        Feature generator instance
    """
    if model_metadata_path:
        # Model-aware generation
        generator = ModelAwareFeatureGenerator(
            dataset=dataset,
            model_metadata_path=model_metadata_path
        )
        return generator
    else:
        # Backward compatible adapter
        return EnhancedFeatureGeneratorAdapter(dataset, selected_features)


# Backward compatibility alias
FeatureGenerator = EnhancedFeatureGeneratorAdapter