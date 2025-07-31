"""
Enhanced Feature Generation System with Registry Pattern
=========================================================

This module provides a robust, model-aware feature generation system that:
- Automatically discovers and registers features
- Handles dependencies between features
- Generates exactly what models need
- Provides clear error handling and validation
"""

import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Set, Callable, Optional, Any, Tuple
from collections import defaultdict, deque
import pandas as pd
from utils.logging_utils import get_logger

logger = get_logger()


@dataclass
class FeatureDefinition:
    """Metadata describing a feature generator function"""
    name: str                           # Generator function name
    func: Callable                      # Generator function
    output_features: List[str]          # Features this generator produces
    dependencies: List[str] = field(default_factory=list)  # Required features/generators
    description: str = ""               # Human-readable description
    category: str = "general"           # Feature category
    
    def __post_init__(self):
        """Validate the feature definition"""
        if not self.output_features:
            raise ValueError(f"Feature {self.name} must specify output_features")


class FeatureRegistry:
    """Central registry for feature generators with dependency resolution"""
    
    def __init__(self):
        self._generators: Dict[str, FeatureDefinition] = {}
        self._feature_to_generators: Dict[str, Set[str]] = defaultdict(set)
        self._category_index: Dict[str, Set[str]] = defaultdict(set)
        
    def register(self, definition: FeatureDefinition) -> None:
        """Register a feature generator"""
        # Store the generator
        self._generators[definition.name] = definition
        
        # Index features to generators
        for feature in definition.output_features:
            self._feature_to_generators[feature].add(definition.name)
        
        # Index by category
        self._category_index[definition.category].add(definition.name)
        
        # logger.debug(f"Registered feature generator '{definition.name}' producing {definition.output_features}")  # Have commented this out as it clutters the logs a bit
    
    def get_available_features(self) -> List[str]:
        """Get all features that can be generated"""
        return list(self._feature_to_generators.keys())
    
    def get_available_generators(self) -> List[str]:
        """Get all registered generator names"""
        return list(self._generators.keys())
    
    def get_generators_for_features(self, feature_names: List[str]) -> Tuple[List[str], List[str]]:
        """
        Get generators needed to produce specific features
        
        Returns:
            (required_generators, missing_features)
        """
        required_generators = set()
        missing_features = []
        
        for feature in feature_names:
            generators = self._feature_to_generators.get(feature, set())
            if generators:
                # For now, take the first generator that can produce this feature
                # In the future, could implement smarter selection logic
                required_generators.add(next(iter(generators)))
            else:
                missing_features.append(feature)
        
        return list(required_generators), missing_features
    
    def resolve_dependencies(self, generator_names: List[str]) -> List[str]:
        """
        Resolve dependencies and return generators in execution order
        Uses topological sort to handle dependencies
        """
        # Build dependency graph
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        all_generators = set(generator_names)
        
        # Add dependencies to the set of required generators
        to_process = deque(generator_names)
        while to_process:
            gen_name = to_process.popleft()
            if gen_name not in self._generators:
                continue
                
            definition = self._generators[gen_name]
            for dep in definition.dependencies:
                # Check if dependency is a feature name that needs to be resolved to a generator
                if dep in self._feature_to_generators:
                    # This is a feature dependency - resolve to its generator(s)
                    dep_generators = self._feature_to_generators[dep]
                    for dep_gen in dep_generators:
                        if dep_gen not in all_generators:
                            all_generators.add(dep_gen)
                            to_process.append(dep_gen)
                        
                        # Add edge: dep_gen -> gen_name
                        graph[dep_gen].append(gen_name)
                        in_degree[gen_name] += 1
                elif dep in self._generators:
                    # This is a generator dependency
                    if dep not in all_generators:
                        all_generators.add(dep)
                        to_process.append(dep)
                    
                    # Add edge: dep -> gen_name
                    graph[dep].append(gen_name)
                    in_degree[gen_name] += 1
                # If dependency is neither a known feature nor generator, skip it
                # (might be a basic column like 'close', 'volume', etc.)
        
        # Initialize in_degree for all generators
        for gen in all_generators:
            if gen not in in_degree:
                in_degree[gen] = 0
        
        # Topological sort
        queue = deque([gen for gen in all_generators if in_degree[gen] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for circular dependencies
        if len(result) != len(all_generators):
            remaining = [gen for gen in all_generators if gen not in result]
            raise ValueError(f"Circular dependency detected among generators: {remaining}")
        
        return result
    
    def get_generator_info(self, generator_name: str) -> Optional[FeatureDefinition]:
        """Get information about a specific generator"""
        return self._generators.get(generator_name)
    
    def validate_feature_availability(self, feature_names: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate that requested features can be generated
        
        Returns:
            (available_features, unavailable_features)
        """
        available = []
        unavailable = []
        
        for feature in feature_names:
            if feature in self._feature_to_generators:
                available.append(feature)
            else:
                unavailable.append(feature)
        
        return available, unavailable
    
    def get_category_generators(self, category: str) -> List[str]:
        """Get all generators in a specific category"""
        return list(self._category_index.get(category, set()))


# Decorator for easy feature registration
def feature_generator(output_features: List[str], 
                     dependencies: List[str] = None, 
                     description: str = "", 
                     category: str = "general"):
    """Decorator to register a method as a feature generator"""
    def decorator(func):
        func._feature_def = FeatureDefinition(
            name=func.__name__,
            func=func,
            output_features=output_features,
            dependencies=dependencies or [],
            description=description,
            category=category
        )
        return func
    return decorator


class AutoRegisterMeta(type):
    """Metaclass that automatically registers decorated methods"""
    def __new__(cls, name, bases, namespace, **kwargs):
        new_class = super().__new__(cls, name, bases, namespace)
        
        # Auto-register methods with feature definitions
        if hasattr(new_class, '_registry'):
            for attr_name in dir(new_class):
                attr = getattr(new_class, attr_name)
                if hasattr(attr, '_feature_def'):
                    new_class._registry.register(attr._feature_def)
        
        return new_class