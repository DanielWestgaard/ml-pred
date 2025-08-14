"""
Data Source Combiners

This module provides components for combining different data formats into unified datasets.
The combiners transform discrete event data (JSON) and continuous time-series data (CSV)
into aligned, feature-rich datasets ready for statistical analysis and ML training.

Key Features:
- Time-indexed feature engineering from discrete events
- Preserves original data storage formats
- Reusable across different data source combinations
- Supports both historical analysis and live trading preparation

Available Combiners:
- TimeSeriesDataCombiner: Main combiner for market data + calendar events
"""

from .base_combiner import BaseCombiner
from .time_series_combiner import TimeSeriesDataCombiner

__all__ = ['BaseCombiner', 'TimeSeriesDataCombiner']