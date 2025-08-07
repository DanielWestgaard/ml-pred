"""
Hypothesis Plugin System for Backtesting

This module provides the plugin architecture for hypothesis-specific customizations
in the backtesting system, enabling rapid testing of different trading strategies
without code modifications.

Directory Structure:
- base_plugin.py: Abstract base class for hypothesis plugins
- exit_strategies.py: Exit strategy implementations and factory
- signal_trackers.py: Signal tracking implementations and factory  
- plugin_registry.py: Plugin discovery and management
- types/: Specific hypothesis plugin implementations
  - spread_compression_plugin.py: Spread compression patterns
  - [future plugins for volume patterns, momentum, etc.]
"""

from .base_plugin import HypothesisPlugin
from .exit_strategies import ExitStrategy, ExitStrategyFactory
from .signal_trackers import SignalTracker, SignalTrackerFactory
from .plugin_registry import HypothesisPluginRegistry

# Import specific hypothesis plugins (automatically registered)
from . import types

__all__ = [
    'HypothesisPlugin',
    'ExitStrategy', 
    'ExitStrategyFactory',
    'SignalTracker',
    'SignalTrackerFactory', 
    'HypothesisPluginRegistry'
]