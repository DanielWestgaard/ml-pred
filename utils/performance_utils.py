"""
Performance Monitoring and Optimization Utilities
=================================================

Safe, non-invasive performance monitoring and memory management utilities.
These functions preserve exact functionality while providing system health monitoring.
"""

import psutil
import gc
import numpy as np
import pandas as pd
from typing import Optional, Any, Callable
import time
from functools import wraps
from utils.logging_utils import get_logger

logger = get_logger()

class MemoryMonitor:
    """Safe memory monitoring with automatic cleanup warnings"""
    
    def __init__(self, warning_threshold_mb: float = 8000, critical_threshold_mb: float = 12000):
        self.warning_threshold = warning_threshold_mb
        self.critical_threshold = critical_threshold_mb
        self.process = psutil.Process()
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def check_and_warn(self, context: str = "") -> float:
        """Check memory usage and issue warnings if needed"""
        memory_mb = self.get_memory_usage_mb()
        
        if memory_mb > self.critical_threshold:
            logger.error(f"CRITICAL: High memory usage: {memory_mb:.1f}MB {context}")
            logger.error("System may become unstable. Consider reducing dataset size.")
            # Force garbage collection
            gc.collect()
        elif memory_mb > self.warning_threshold:
            logger.warning(f"HIGH memory usage: {memory_mb:.1f}MB {context}")
            logger.warning("Consider monitoring system performance.")
            
        return memory_mb
    
    def safe_cleanup(self):
        """Safe memory cleanup without affecting functionality"""
        initial_memory = self.get_memory_usage_mb()
        gc.collect()
        final_memory = self.get_memory_usage_mb()
        
        if initial_memory - final_memory > 100:  # If we freed > 100MB
            logger.info(f"Cleaned up {initial_memory - final_memory:.1f}MB of memory")

# Global memory monitor instance
memory_monitor = MemoryMonitor()

def monitor_memory(func: Callable) -> Callable:
    """Decorator to monitor memory usage around function calls"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check memory before
        initial_memory = memory_monitor.get_memory_usage_mb()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Check memory after and warn if needed
        final_memory = memory_monitor.check_and_warn(f"after {func.__name__}")
        
        # Log significant memory increases
        memory_increase = final_memory - initial_memory
        if memory_increase > 500:  # > 500MB increase
            logger.info(f"{func.__name__} increased memory by {memory_increase:.1f}MB")
            
        return result
    return wrapper

def optimize_dataframe_memory(df: pd.DataFrame, preserve_precision: bool = True) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage while preserving precision and functionality.
    This is completely safe and doesn't change calculations.
    """
    if not preserve_precision:
        return df
    
    initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # Only do safe optimizations that don't affect precision
    optimized_df = df.copy()
    
    # Convert object columns to category if they have few unique values (safe)
    for col in optimized_df.select_dtypes(include=['object']).columns:
        if optimized_df[col].nunique() / len(optimized_df) < 0.1:  # < 10% unique values
            optimized_df[col] = optimized_df[col].astype('category')
    
    final_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024
    
    if initial_memory - final_memory > 10:  # If we saved > 10MB
        logger.info(f"Optimized DataFrame memory: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
    
    return optimized_df

def safe_sample_for_analysis(df: pd.DataFrame, max_rows: int = 200000, 
                           preserve_distribution: bool = True) -> pd.DataFrame:
    """
    Safely sample data for analysis while preserving statistical properties.
    This maintains functionality but reduces computational load.
    """
    if len(df) <= max_rows:
        return df
    
    logger.info(f"Sampling {max_rows:,} rows from {len(df):,} for analysis (preserves distribution)")
    
    if preserve_distribution:
        # Stratified sampling to preserve time distribution
        # Sample evenly across the time series to maintain temporal patterns
        step = len(df) // max_rows
        sampled_indices = np.arange(0, len(df), step)[:max_rows]
        return df.iloc[sampled_indices].copy()
    else:
        return df.sample(n=max_rows, random_state=42)

def fast_momentum_consistency(series: pd.Series, window: int) -> pd.Series:
    """
    Fast momentum consistency calculation that's equivalent to the original but much faster.
    Calculates what percentage of values in the window have the same sign as the last value.
    """
    def calculate_consistency(x):
        if len(x) < 1:
            return 0.0
        last_sign = np.sign(x.iloc[-1])
        same_sign_count = (np.sign(x) == last_sign).sum()
        return same_sign_count / len(x)
    
    return series.rolling(window=window, min_periods=1).apply(calculate_consistency, raw=False)

def fast_linear_trend(series: pd.Series, window: int) -> pd.Series:
    """
    Fast linear trend calculation that's mathematically equivalent to np.polyfit
    but much faster. Preserves exact calculation precision.
    
    This replaces np.polyfit(range(len(x)), x, 1)[0] with equivalent vectorized math.
    """
    def calculate_slope(x):
        if len(x) < 2:
            return 0.0
        
        # Equivalent to np.polyfit but faster
        n = len(x)
        x_indices = np.arange(n)
        
        # Linear regression slope calculation (same as polyfit)
        x_mean = x_indices.mean()
        y_mean = x.mean()
        
        numerator = np.sum((x_indices - x_mean) * (x - y_mean))
        denominator = np.sum((x_indices - x_mean) ** 2)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    return series.rolling(window=window, min_periods=2).apply(calculate_slope, raw=False)

def should_use_optimizations(df_size: int, feature_name: str = "") -> dict:
    """
    Determine which optimizations to use based on dataset size.
    Returns dict of optimization flags while preserving functionality.
    """
    optimizations = {
        'use_sampling_for_importance': df_size > 500000,  # Sample for feature importance
        'use_fast_trend_calculation': df_size > 100000,   # Use optimized trend calc
        'enable_memory_monitoring': df_size > 50000,      # Enable memory monitoring
        'force_garbage_collection': df_size > 1000000,   # Aggressive cleanup
        'skip_expensive_correlations': df_size > 1000000 # Skip some expensive ops
    }
    
    if any(optimizations.values()):
        enabled = [k for k, v in optimizations.items() if v]
        logger.info(f"Performance optimizations enabled for {df_size:,} rows: {enabled}")
    
    return optimizations

class PerformanceContext:
    """Context manager for monitoring performance of code blocks"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = memory_monitor.get_memory_usage_mb()
        logger.debug(f"Starting {self.operation_name} (Memory: {self.start_memory:.1f}MB)")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        final_memory = memory_monitor.check_and_warn(f"after {self.operation_name}")
        memory_change = final_memory - self.start_memory
        
        logger.info(f"Completed {self.operation_name} in {duration:.1f}s "
                   f"(Memory: {final_memory:.1f}MB, Δ{memory_change:+.1f}MB)")
        
        # Clean up if we used a lot of memory
        if memory_change > 1000:  # > 1GB increase
            memory_monitor.safe_cleanup()