#!/usr/bin/env python3
"""
Signal-Rich Period Identification Script
========================================

Systematically identify periods with high signal activity for the spread compression model.
This script analyzes historical data to find optimal backtesting periods across different months.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
import sys
sys.path.append('/Users/danielwestgaard/progRel/ml-pred')

from data.features.enhanced_generation import ModelAwareFeatureGenerator
from utils.logging_utils import setup_logging

def load_model_and_features():
    """Load the trained spread compression model and get feature names."""
    model_path = Path('/Users/danielwestgaard/progRel/ml-pred/storage/models/2025-07-27_12:53:02/random_forest.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Model loaded with {len(model.feature_names_in_)} features")
    return model

def analyze_signal_distribution(data_path: str, model, sample_days: int = 30):
    """
    Analyze signal distribution across different periods.
    
    Args:
        data_path: Path to the processed data file
        model: Trained model
        sample_days: Number of days to analyze per period
    """
    print(f"📊 Analyzing signal distribution in {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # Feature generator will be initialized for each period
    
    # Define analysis periods (different months and market conditions)
    analysis_periods = []
    
    # Get available date range
    start_date = df.index.min()
    end_date = df.index.max()
    print(f"📅 Data available from {start_date} to {end_date}")
    
    # Generate monthly periods for analysis
    current = start_date.replace(day=1)
    while current + timedelta(days=sample_days) < end_date:
        period_start = current
        period_end = current + timedelta(days=sample_days)
        
        # Skip periods without enough data
        period_data = df[period_start:period_end]
        if len(period_data) > 24 * 60 * sample_days * 0.8:  # At least 80% data coverage
            analysis_periods.append({
                'name': f"{period_start.strftime('%Y-%m')}_sample",
                'start': period_start,
                'end': period_end,
                'market_condition': 'unknown'  # We'll classify this later
            })
        
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    print(f"🔍 Analyzing {len(analysis_periods)} periods")
    
    # Analyze each period
    signal_results = []
    
    for i, period in enumerate(analysis_periods):
        print(f"\n📈 Analyzing period {i+1}/{len(analysis_periods)}: {period['name']}")
        
        try:
            # Extract period data
            period_data = df[period['start']:period['end']].copy()
            
            if len(period_data) < 1000:  # Skip periods with insufficient data
                print(f"⚠️  Skipping {period['name']} - insufficient data ({len(period_data)} rows)")
                continue
            
            # Use existing features (data is already processed)
            print("   🔧 Extracting model features...")
            
            # Get model features that exist in the data (excluding target columns)
            prediction_features = [f for f in model.feature_names_in_ 
                                 if f in period_data.columns and 
                                 not f.startswith('target_') and 
                                 not f.startswith('future_')]
            
            if len(prediction_features) < 8:  # Need minimum features for meaningful predictions
                print(f"⚠️  Skipping {period['name']} - insufficient prediction features ({len(prediction_features)} available)")
                continue
            
            # Extract model features
            model_features = period_data[prediction_features].dropna()
            
            if len(model_features) < 100:  # Skip if not enough valid features
                print(f"⚠️  Skipping {period['name']} - insufficient feature data ({len(model_features)} rows)")
                continue
            
            # Get model predictions
            print("   🤖 Getting model predictions...")
            predictions = model.predict(model_features)
            probabilities = model.predict_proba(model_features)
            
            # Calculate signal metrics
            positive_signals = np.sum(predictions == 1)
            total_predictions = len(predictions)
            signal_rate = positive_signals / total_predictions if total_predictions > 0 else 0
            
            # High-confidence signals (>0.8 probability)
            high_conf_signals = np.sum(probabilities[:, 1] > 0.8) if probabilities.shape[1] > 1 else 0
            high_conf_rate = high_conf_signals / total_predictions if total_predictions > 0 else 0
            
            # Average probability for positive class
            avg_positive_prob = np.mean(probabilities[:, 1]) if probabilities.shape[1] > 1 else 0
            max_positive_prob = np.max(probabilities[:, 1]) if probabilities.shape[1] > 1 else 0
            
            # Market condition analysis
            price_volatility = enhanced_data['close'].std() if 'close' in enhanced_data.columns else 0
            price_change = (enhanced_data['close'].iloc[-1] - enhanced_data['close'].iloc[0]) / enhanced_data['close'].iloc[0] if 'close' in enhanced_data.columns else 0
            
            # Classification of market condition
            market_condition = "unknown"
            if abs(price_change) > 0.02:  # >2% change
                market_condition = "trending"
            elif price_volatility > enhanced_data['close'].mean() * 0.01:  # High volatility
                market_condition = "volatile"
            else:
                market_condition = "sideways"
            
            result = {
                'period_name': period['name'],
                'start_date': period['start'].strftime('%Y-%m-%d'),
                'end_date': period['end'].strftime('%Y-%m-%d'),
                'total_data_points': len(period_data),
                'valid_predictions': total_predictions,
                'positive_signals': int(positive_signals),
                'signal_rate': signal_rate,
                'high_conf_signals': int(high_conf_signals),
                'high_conf_rate': high_conf_rate,
                'avg_positive_prob': avg_positive_prob,
                'max_positive_prob': max_positive_prob,
                'price_volatility': price_volatility,
                'price_change_pct': price_change * 100,
                'market_condition': market_condition,
                'signals_per_day': (positive_signals / sample_days) if sample_days > 0 else 0,
                'high_conf_per_day': (high_conf_signals / sample_days) if sample_days > 0 else 0
            }
            
            signal_results.append(result)
            
            print(f"   ✅ {positive_signals} signals ({signal_rate:.1%}) | {high_conf_signals} high-conf ({high_conf_rate:.1%}) | {market_condition}")
            
        except Exception as e:
            print(f"   ❌ Error analyzing {period['name']}: {str(e)}")
            continue
    
    return signal_results

def rank_periods_for_backtesting(signal_results):
    """Rank periods by backtesting value."""
    
    df_results = pd.DataFrame(signal_results)
    
    if len(df_results) == 0:
        print("❌ No valid periods found for analysis")
        return []
    
    # Calculate composite score for backtesting value
    # Prioritize periods with:
    # 1. Good signal frequency (not too sparse, not too dense)
    # 2. High confidence signals
    # 3. Diverse market conditions
    
    df_results['signal_quality_score'] = (
        df_results['high_conf_rate'] * 0.4 +  # 40% weight on high confidence
        np.clip(df_results['signal_rate'] * 50, 0, 1) * 0.3 +  # 30% weight on reasonable signal rate (optimal around 2%)
        (df_results['avg_positive_prob'] - 0.5) * 0.3  # 30% weight on average confidence
    )
    
    # Market condition diversity bonus
    condition_counts = df_results['market_condition'].value_counts()
    df_results['condition_rarity_bonus'] = df_results['market_condition'].map(
        lambda x: 1.0 / condition_counts[x] if x in condition_counts else 0
    ) * 0.1
    
    df_results['backtest_score'] = (
        df_results['signal_quality_score'] + 
        df_results['condition_rarity_bonus']
    )
    
    # Sort by backtest score
    df_ranked = df_results.sort_values('backtest_score', ascending=False)
    
    return df_ranked

def save_results_and_configs(ranked_periods, output_dir):
    """Save analysis results and generate backtest configs."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save full analysis results
    analysis_file = output_path / 'signal_rich_periods_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(ranked_periods.to_dict('records'), f, indent=2)
    
    print(f"📋 Saved analysis results to {analysis_file}")
    
    # Generate summary report
    report_file = output_path / 'signal_analysis_summary.md'
    
    with open(report_file, 'w') as f:
        f.write("# Signal-Rich Period Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Top 10 Periods for Backtesting\n\n")
        f.write("| Rank | Period | Signal Rate | High-Conf Rate | Signals/Day | Market Condition | Score |\n")
        f.write("|------|--------|-------------|----------------|-------------|------------------|-------|\n")
        
        for i, row in ranked_periods.head(10).iterrows():
            f.write(f"| {ranked_periods.index.get_loc(i)+1} | {row['period_name']} | {row['signal_rate']:.1%} | "
                   f"{row['high_conf_rate']:.1%} | {row['signals_per_day']:.1f} | {row['market_condition']} | "
                   f"{row['backtest_score']:.3f} |\n")
        
        f.write("\n## Market Condition Distribution\n\n")
        condition_counts = ranked_periods['market_condition'].value_counts()
        for condition, count in condition_counts.items():
            f.write(f"- **{condition.title()}**: {count} periods ({count/len(ranked_periods):.1%})\n")
        
        f.write("\n## Signal Quality Statistics\n\n")
        f.write(f"- **Average Signal Rate**: {ranked_periods['signal_rate'].mean():.2%}\n")
        f.write(f"- **Average High-Confidence Rate**: {ranked_periods['high_conf_rate'].mean():.2%}\n")
        f.write(f"- **Average Signals per Day**: {ranked_periods['signals_per_day'].mean():.1f}\n")
        f.write(f"- **Best Signal Rate**: {ranked_periods['signal_rate'].max():.2%} ({ranked_periods.loc[ranked_periods['signal_rate'].idxmax(), 'period_name']})\n")
    
    print(f"📊 Saved summary report to {report_file}")
    
    # Generate config files for top periods
    config_template_path = Path('/Users/danielwestgaard/progRel/ml-pred/hypothesis/spread_compression_patterns/config_backtest.yml')
    with open(config_template_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    top_periods = ranked_periods.head(15)  # Top 15 for systematic testing
    config_dir = output_path / 'backtest_configs'
    config_dir.mkdir(exist_ok=True)
    
    for i, row in top_periods.iterrows():
        config = base_config.copy()
        
        # Update config for this period
        config['backtesting']['start_date'] = row['start_date']
        config['backtesting']['end_date'] = row['end_date']
        
        # Adjust max_loops based on expected signal frequency
        expected_signals = row['signals_per_day'] * 30  # Assuming ~30 day period
        config['backtesting']['execution']['max_loops'] = max(2000, int(expected_signals * 100))  # Allow plenty of room
        
        # Adjust confidence threshold based on signal quality
        if row['high_conf_rate'] > 0.5:  # If many high-confidence signals, be more selective
            config['backtesting']['signal_processing']['confidence_threshold'] = 0.70
        else:
            config['backtesting']['signal_processing']['confidence_threshold'] = 0.50
        
        # Save config
        config_filename = f"config_backtest_{row['period_name']}.yml"
        config_path = config_dir / config_filename
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    print(f"⚙️  Generated {len(top_periods)} backtest configs in {config_dir}")
    
    return top_periods

def main():
    """Main execution function."""
    print("🔍 Starting Signal-Rich Period Identification")
    print("=" * 50)
    
    logger = setup_logging("signal_analysis", "mixed")
    
    # Load model
    model = load_model_and_features()
    
    # Data path (using the main training dataset)
    data_path = '/Users/danielwestgaard/progRel/ml-pred/storage/data/capital_com/GBPUSD/processed/1m/2024-06-01_2025-05-23.csv'
    
    # Analyze signal distribution
    signal_results = analyze_signal_distribution(data_path, model, sample_days=30)
    
    if not signal_results:
        print("❌ No periods found with sufficient signal data")
        return
    
    # Rank periods
    ranked_periods = rank_periods_for_backtesting(signal_results)
    
    print(f"\n🏆 TOP 10 SIGNAL-RICH PERIODS:")
    print("-" * 80)
    for i, row in ranked_periods.head(10).iterrows():
        print(f"{ranked_periods.index.get_loc(i)+1:2d}. {row['period_name']:15s} | "
              f"Signals: {row['signal_rate']:6.1%} ({row['signals_per_day']:4.1f}/day) | "
              f"High-Conf: {row['high_conf_rate']:6.1%} | "
              f"{row['market_condition']:9s} | Score: {row['backtest_score']:.3f}")
    
    # Save results and generate configs
    output_dir = '/Users/danielwestgaard/progRel/ml-pred/results/signal_analysis'
    top_periods = save_results_and_configs(ranked_periods, output_dir)
    
    print(f"\n✅ Analysis complete! Found {len(signal_results)} valid periods.")
    print(f"📂 Results saved to: {output_dir}")
    print(f"🎯 Ready to run systematic backtests on top {len(top_periods)} periods.")

if __name__ == "__main__":
    main()