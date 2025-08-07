"""
Base Target Generator with Universal Feature Dependency System
============================================================

This module provides a base class for all target hypothesis generators that:
- Handles universal feature dependency resolution
- Manages temporary feature generation and cleanup
- Provides consistent validation and error handling
- Integrates seamlessly with the existing FeatureRegistry system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from abc import ABC, abstractmethod

from data.processing.base_processor import BaseProcessor
from data.features.enhanced_generation import ModelAwareFeatureGenerator
from utils import data_utils
from utils.logging_utils import get_logger

logger = get_logger()


class BaseTargetGenerator(BaseProcessor, ABC):
    """
    Base class for all target hypothesis generators.
    
    Provides universal feature dependency resolution, temporary feature generation,
    and consistent cleanup patterns. All hypothesis-specific target generators
    should inherit from this class.
    """
    
    def __init__(self, dataset, hypothesis: str):
        """
        Initialize base target generator.
        
        Args:
            dataset: Processed data with specific features generated
            hypothesis: The hypothesis name to test
        """
        try:
            # Load dataset based on format
            self.df, self.original_df = data_utils.check_and_return_df(dataset)
            self.hypothesis = hypothesis
            self._temp_generated_features = []
            self._temp_generator = None
            
        except Exception as e:
            logger.error(f"Unable to initialize Base Target Generator: {e}")
            raise
    
    def run(self) -> pd.DataFrame:
        """
        Run target generation with universal error handling and cleanup.
        
        Returns:
            DataFrame with generated target column(s)
        """
        logger.info(f"---------- Running Hypothesis Targeting: {self.hypothesis} ----------")
        
        try:
            # Check if the hypothesis method exists
            if not hasattr(self, self.hypothesis):
                available_methods = [method for method in dir(self) 
                                   if not method.startswith('_') and 
                                   callable(getattr(self, method)) and
                                   method not in ['run', 'generate_features_with_dependencies', 'cleanup_temp_features']]
                logger.error(f"Could not find method '{self.hypothesis}' in {self.__class__.__name__}")
                logger.error(f"Available hypothesis methods: {available_methods}")
                return self.df
            
            # Execute the hypothesis-specific target generation
            success = getattr(self, self.hypothesis)()
            
            if not success:
                logger.error(f"Target generation failed for hypothesis: {self.hypothesis}")
            else:
                logger.info(f"✅ Target generation completed successfully for: {self.hypothesis}")
                
        except Exception as e:
            logger.error(f"Error during target generation for {self.hypothesis}: {e}")
            
        finally:
            # Always cleanup temporary features
            self.cleanup_temp_features()
            
        return self.df
    
    def generate_features_with_dependencies(self, 
                                          required_features: List[str], 
                                          feature_dependency_map: Optional[Dict[str, Tuple[str, List[str]]]] = None
                                          ) -> Tuple[List[str], List[str]]:
        """
        Universal method to generate required features and their dependencies.
        
        This method replaces the duplicated _generate_features_with_dependencies
        logic from the original target_hypothesis.py file.
        
        Args:
            required_features: List of feature names that are required
            feature_dependency_map: Optional dict mapping feature_name -> (generator_method, [dependencies])
                                   If None, uses the global feature registry
        
        Returns:
            tuple: (successfully_generated_features, all_added_features)
        """
        missing_features = [f for f in required_features if f not in self.df.columns]
        if not missing_features:
            logger.debug("All required features already present")
            return [], []
            
        logger.warning(f"Missing required features for {self.hypothesis}: {missing_features}")
        logger.info("Generating missing features temporarily...")
        
        # Initialize temporary generator if not already done
        if self._temp_generator is None:
            self._temp_generator = ModelAwareFeatureGenerator(self.df)
        
        # Track original columns to identify what was added
        original_columns = set(self._temp_generator.df.columns)
        
        # Ensure datetime index for session features
        if 'date' in self._temp_generator.df.columns:
            self._temp_generator.df.set_index('date', inplace=True)
            self._temp_generator.df.index = pd.to_datetime(self._temp_generator.df.index)
        
        successfully_generated = []
        
        if feature_dependency_map:
            # Use provided dependency mapping (legacy support)
            successfully_generated = self._generate_with_custom_mapping(
                missing_features, feature_dependency_map, original_columns
            )
        else:
            # Use the global feature registry (preferred approach)
            successfully_generated = self._generate_with_registry(
                missing_features, original_columns
            )
        
        # Update our dataframe and track all added features
        self.df = self._temp_generator.df
        final_columns = set(self._temp_generator.df.columns)
        all_added_features = list(final_columns - original_columns)
        
        # Store for cleanup
        self._temp_generated_features.extend(all_added_features)
        
        logger.info(f"Successfully generated {len(successfully_generated)} features: {successfully_generated}")
        logger.debug(f"Total features added: {len(all_added_features)}")
        
        return successfully_generated, all_added_features
    
    def _generate_with_custom_mapping(self, 
                                    missing_features: List[str], 
                                    feature_dependency_map: Dict[str, Tuple[str, List[str]]], 
                                    original_columns: Set[str]) -> List[str]:
        """Generate features using custom dependency mapping (legacy support)."""
        
        # Build complete dependency graph including transitive dependencies
        all_needed_features = set(missing_features)
        
        def add_dependencies_recursively(feature_name):
            if feature_name in feature_dependency_map:
                _, dependencies = feature_dependency_map[feature_name]
                if dependencies:
                    for dep in dependencies:
                        if dep not in self._temp_generator.df.columns:
                            all_needed_features.add(dep)
                            add_dependencies_recursively(dep)
        
        for feature in missing_features:
            add_dependencies_recursively(feature)
        
        # Topologically sort features based on dependencies
        generation_order = self._topological_sort_custom(all_needed_features, feature_dependency_map)
        successfully_generated = []
        
        # Generate features in dependency order
        for feature_name in generation_order:
            if feature_name not in self._temp_generator.df.columns:
                if feature_name in feature_dependency_map:
                    generator_method, _ = feature_dependency_map[feature_name]
                    if generator_method and hasattr(self._temp_generator, generator_method):
                        try:
                            logger.debug(f"Generating {feature_name} using {generator_method}")
                            getattr(self._temp_generator, generator_method)()
                            if feature_name in self._temp_generator.df.columns:
                                successfully_generated.append(feature_name)
                            else:
                                logger.warning(f"Feature {feature_name} not found after generation")
                        except Exception as e:
                            logger.warning(f"Failed to generate {feature_name}: {e}")
                    else:
                        logger.warning(f"Cannot generate {feature_name}: no method {generator_method}")
        
        return successfully_generated
    
    def _generate_with_registry(self, missing_features: List[str], original_columns: Set[str]) -> List[str]:
        """Generate features using the global feature registry (preferred approach)."""
        
        # Get generators needed for missing features
        required_generators, unavailable_features = self._temp_generator._registry.get_generators_for_features(missing_features)
        
        if unavailable_features:
            logger.warning(f"Cannot generate features (not in registry): {unavailable_features}")
        
        if not required_generators:
            logger.warning("No generators found for missing features")
            return []
        
        # Resolve dependencies and get execution order
        try:
            execution_order = self._temp_generator._registry.resolve_dependencies(required_generators)
            logger.debug(f"Feature generation order: {execution_order}")
        except ValueError as e:
            logger.error(f"Dependency resolution failed: {e}")
            return []
        
        successfully_generated = []
        
        # Execute generators in dependency order
        for generator_name in execution_order:
            if hasattr(self._temp_generator, generator_name):
                try:
                    logger.debug(f"Executing generator: {generator_name}")
                    getattr(self._temp_generator, generator_name)()
                    
                    # Check what features were generated
                    current_columns = set(self._temp_generator.df.columns)
                    newly_generated = list(current_columns - original_columns - set(successfully_generated))
                    
                    if newly_generated:
                        successfully_generated.extend(newly_generated)
                        logger.debug(f"Generator {generator_name} produced: {newly_generated}")
                    
                except Exception as e:
                    logger.warning(f"Failed to execute generator {generator_name}: {e}")
            else:
                logger.warning(f"Generator method {generator_name} not found in ModelAwareFeatureGenerator")
        
        return successfully_generated
    
    def _topological_sort_custom(self, 
                                features: Set[str], 
                                dependency_map: Dict[str, Tuple[str, List[str]]]) -> List[str]:
        """Topologically sort features for custom dependency mapping."""
        
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
            
            if feature in dependency_map:
                _, dependencies = dependency_map[feature]
                if dependencies:
                    for dep in dependencies:
                        if dep in features:
                            visit(dep)
            
            temp_visited.remove(feature)
            visited.add(feature)
            result.append(feature)
        
        for feature in features:
            if feature not in visited:
                visit(feature)
        
        return result
    
    def cleanup_temp_features(self):
        """Clean up all temporarily generated features."""
        if self._temp_generated_features and len(self._temp_generated_features) > 0:
            # Only cleanup features that actually exist in the dataframe
            features_to_cleanup = [f for f in self._temp_generated_features if f in self.df.columns]
            
            if features_to_cleanup:
                logger.info(f"Cleaning up {len(features_to_cleanup)} temporary features")
                logger.debug(f"Cleaning up: {features_to_cleanup}")
                self.df = self.df.drop(columns=features_to_cleanup)
            
            # Clear the tracking list
            self._temp_generated_features = []
        
        # Reset temporary generator
        self._temp_generator = None
    
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """
        Get the list of features required by this hypothesis.
        
        This should be implemented by each hypothesis to define what features it needs.
        Can be static or dynamic based on configuration.
        
        Returns:
            List of required feature names
        """
        pass
    
    @abstractmethod
    def get_feature_dependencies(self) -> Optional[Dict[str, Tuple[str, List[str]]]]:
        """
        Get custom feature dependency mapping for this hypothesis.
        
        Override this method if the hypothesis needs custom dependency logic
        that isn't covered by the global feature registry.
        
        Returns:
            Optional dict mapping feature_name -> (generator_method, [dependencies])
            None to use global feature registry
        """
        return None
    
    # Abstract method placeholder - subclasses will implement their specific target generation
    # The specific hypothesis methods (e.g., spread_compression_patterns_v4) will be implemented
    # in the concrete subclasses