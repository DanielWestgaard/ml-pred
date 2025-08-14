"""
Combination Testing Framework for Multi-Data Sources

This module provides systematic testing of different data source combinations
to identify market inefficiencies that institutions miss due to organizational
constraints and scale limitations.

Core Components:
- CombinationTestOrchestrator: Main entry point for combination testing
- StatisticalTestStrategy: Basic statistical analysis of combinations
- Results storage and analysis framework
"""

from .combination_orchestrator import CombinationTestOrchestrator

__all__ = ['CombinationTestOrchestrator']