"""
Hypothesis-Specific Plugin Implementations

This module contains concrete implementations of hypothesis plugins
for different trading strategies and patterns.
"""

from .spread_compression_plugin import SpreadCompressionPlugin

__all__ = [
    'SpreadCompressionPlugin'
]