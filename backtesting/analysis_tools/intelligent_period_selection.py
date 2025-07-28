#!/usr/bin/env python3
"""
Intelligent Period Selection
============================

Select optimal testing periods based on market conditions rather than arbitrary dates.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def identify_optimal_periods(data_path, target_days_per_period=7):
    """Identify periods with optimal characteristics for spread compression testing."""
    
    # Load data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # Calculate rolling metrics for period selection
    window = target_days_per_period * 24 * 60  # Convert days to minutes
    
    df['volatility'] = df['close'].rolling(window).std()
    df['signal_density'] = df['target_spread_compression_binary'].rolling(window).sum()
    df['london_session_pct'] = df['london_session'].rolling(window).mean()
    df['volume_activity'] = df['volume'].rolling(window).mean()
    
    # Score each potential period
    df['period_score'] = (
        (df['volatility'] / df['volatility'].median()) * 0.3 +  # 30% weight on volatility
        (df['signal_density'] / 10) * 0.4 +  # 40% weight on signal density
        df['london_session_pct'] * 0.2 +  # 20% weight on London session coverage
        (df['volume_activity'] / df['volume_activity'].median()) * 0.1  # 10% weight on volume
    )
    
    # Find top periods
    optimal_periods = []
    
    # Use non-overlapping windows
    current_date = df.index.min()
    end_date = df.index.max() - timedelta(days=target_days_per_period)
    
    while current_date < end_date:
        period_end = current_date + timedelta(days=target_days_per_period)
        period_data = df[current_date:period_end]
        
        if len(period_data) > window * 0.8:  # At least 80% data coverage
            avg_score = period_data['period_score'].mean()
            signal_count = period_data['target_spread_compression_binary'].sum()
            
            optimal_periods.append({
                'start_date': current_date,
                'end_date': period_end,
                'score': avg_score,
                'signal_count': signal_count,
                'volatility': period_data['volatility'].mean(),
                'london_session_pct': period_data['london_session_pct'].mean(),
            })
        
        current_date += timedelta(days=target_days_per_period // 2)  # 50% overlap for sliding window
    
    # Sort by score and return top periods
    optimal_df = pd.DataFrame(optimal_periods)
    optimal_df = optimal_df.sort_values('score', ascending=False)
    
    return optimal_df.head(10)  # Return top 10 periods

if __name__ == "__main__":
    data_path = '/Users/danielwestgaard/progRel/ml-pred/storage/data/capital_com/GBPUSD/processed/1m/2024-06-01_2025-05-23.csv'
    optimal_periods = identify_optimal_periods(data_path)
    
    print("🎯 TOP 10 OPTIMAL PERIODS FOR TESTING:")
    print("=" * 60)
    for i, row in optimal_periods.iterrows():
        print(f"{i+1:2d}. {row['start_date'].strftime('%Y-%m-%d')} to {row['end_date'].strftime('%Y-%m-%d')}")
        print(f"    Score: {row['score']:.3f} | Signals: {row['signal_count']:3.0f} | "
              f"Volatility: {row['volatility']:.6f} | London %: {row['london_session_pct']:.1%}")
        print()
