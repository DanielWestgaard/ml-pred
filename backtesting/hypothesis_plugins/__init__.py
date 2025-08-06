"""
Hypothesis Plugin System for Backtesting

This module provides the plugin architecture for hypothesis-specific customizations
in the backtesting system, enabling rapid testing of different trading strategies
without code modifications.
"""

from .base_plugin import HypothesisPlugin
from .exit_strategies import ExitStrategy, ExitStrategyFactory
from .signal_trackers import SignalTracker, SignalTrackerFactory
from .plugin_registry import HypothesisPluginRegistry

__all__ = [
    'HypothesisPlugin',
    'ExitStrategy', 
    'ExitStrategyFactory',
    'SignalTracker',
    'SignalTrackerFactory', 
    'HypothesisPluginRegistry'
]