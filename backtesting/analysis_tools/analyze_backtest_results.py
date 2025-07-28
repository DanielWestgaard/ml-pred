#!/usr/bin/env python3
"""
Backtest Results Analysis
========================

Analyze the results from multiple backtests to:
1. Optimize risk parameters
2. Compare performance across different market conditions
3. Analyze model's exit timing vs manual stop loss/take profit
4. Identify signal-rich periods and patterns
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_all_backtest_results():
    """Load all available backtest results."""
    results_dir = Path('/Users/danielwestgaard/progRel/ml-pred/storage/models/2025-07-27_12:53:02/backtesting')
    
    results = []
    result_files = list(results_dir.glob('backtest_results_*.json'))
    
    print(f"🔍 Found {len(result_files)} backtest result files")
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                result['file_name'] = file_path.name
                result['timestamp'] = file_path.name.split('_')[-1].replace('.json', '')
                results.append(result)
        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")
    
    return results

def extract_analysis_data(results):
    """Extract key metrics from backtest results for analysis."""
    
    analysis_data = []
    
    for result in results:
        backtest_summary = result.get('backtest_summary', {})
        trading_perf = result.get('trading_performance', {})
        risk_metrics = result.get('risk_metrics', {})
        trade_analysis = result.get('trade_analysis', {})
        data_period = result.get('data_period', {})
        
        # Parse start/end dates if available
        start_date = data_period.get('start', 'Unknown')
        end_date = data_period.get('end', 'Unknown')
        
        # Calculate period length
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            period_days = (end_dt - start_dt).days
        except:
            period_days = 0
        
        # Extract exit reasons if available
        exit_reasons = trade_analysis.get('exit_reasons', {})
        model_exits = exit_reasons.get('model_exit_signal', 0)
        stop_loss_exits = exit_reasons.get('stop_loss', 0)
        take_profit_exits = exit_reasons.get('take_profit', 0)
        
        analysis_data.append({
            'timestamp': result.get('timestamp', 'unknown'),
            'period_start': start_date,
            'period_end': end_date,
            'period_days': period_days,
            'total_return_pct': backtest_summary.get('total_return_pct', 0),
            'initial_capital': backtest_summary.get('initial_capital', 10000),
            'final_equity': backtest_summary.get('final_equity', 10000),
            'execution_time_seconds': backtest_summary.get('execution_time_seconds', 0),
            
            # Trading Performance
            'total_trades': trading_perf.get('total_trades', 0),
            'winning_trades': trading_perf.get('winning_trades', 0),
            'losing_trades': trading_perf.get('losing_trades', 0),
            'win_rate': trading_perf.get('win_rate', 0),
            'total_fees': trading_perf.get('total_fees', 0),
            'max_drawdown_pct': trading_perf.get('max_drawdown_pct', 0),
            
            # Risk Metrics
            'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
            'sortino_ratio': risk_metrics.get('sortino_ratio', 0),
            'profit_factor': risk_metrics.get('profit_factor', 0),
            'average_win': risk_metrics.get('average_win', 0),
            'average_loss': risk_metrics.get('average_loss', 0),
            'largest_win': risk_metrics.get('largest_win', 0),
            'largest_loss': risk_metrics.get('largest_loss', 0),
            
            # Trade Analysis
            'avg_holding_period_minutes': trade_analysis.get('average_holding_period_minutes', 0),
            'median_holding_period_minutes': trade_analysis.get('median_holding_period_minutes', 0),
            'avg_pnl_per_trade': trade_analysis.get('average_pnl_per_trade', 0),
            'median_pnl_per_trade': trade_analysis.get('median_pnl_per_trade', 0),
            'pnl_std': trade_analysis.get('pnl_std', 0),
            'consecutive_wins_max': trade_analysis.get('consecutive_wins_max', 0),
            'consecutive_losses_max': trade_analysis.get('consecutive_losses_max', 0),
            
            # Exit Analysis
            'model_exits': model_exits,
            'stop_loss_exits': stop_loss_exits,
            'take_profit_exits': take_profit_exits,
            'model_exit_pct': (model_exits / max(trading_perf.get('total_trades', 1), 1)) * 100,
            'stop_loss_pct': (stop_loss_exits / max(trading_perf.get('total_trades', 1), 1)) * 100,
            'take_profit_pct': (take_profit_exits / max(trading_perf.get('total_trades', 1), 1)) * 100,
            
            # Derived Metrics
            'trades_per_day': trading_perf.get('total_trades', 0) / max(period_days, 1),
            'profit_per_day': backtest_summary.get('total_return_pct', 0) / max(period_days, 1),
            'fees_as_pct_of_capital': (trading_perf.get('total_fees', 0) / backtest_summary.get('initial_capital', 10000)) * 100,
        })
    
    return pd.DataFrame(analysis_data)

def analyze_exit_timing_patterns(df):
    """Analyze model exit timing vs manual stop loss/take profit effectiveness."""
    
    print("\n🎯 EXIT TIMING ANALYSIS")
    print("=" * 50)
    
    # Overall exit strategy effectiveness
    print(f"📊 Exit Strategy Distribution:")
    print(f"   • Model exits: {df['model_exit_pct'].mean():.1f}% average")
    print(f"   • Stop loss exits: {df['stop_loss_pct'].mean():.1f}% average") 
    print(f"   • Take profit exits: {df['take_profit_pct'].mean():.1f}% average")
    
    # Analyze if model exits are too aggressive
    model_dominant = df[df['model_exit_pct'] > 80]
    print(f"\n🤖 Model-Dominant Periods ({len(model_dominant)} tests):")
    if len(model_dominant) > 0:
        print(f"   • Average return: {model_dominant['total_return_pct'].mean():.2f}%")
        print(f"   • Average win rate: {model_dominant['win_rate'].mean():.1%}")
        print(f"   • Average holding period: {model_dominant['avg_holding_period_minutes'].mean():.1f} min")
        print(f"   • Average trades per day: {model_dominant['trades_per_day'].mean():.1f}")
    
    # Analyze if we need longer holding periods
    print(f"\n⏱️  Holding Period Analysis:")
    print(f"   • Average holding time: {df['avg_holding_period_minutes'].mean():.1f} minutes")
    print(f"   • Median holding time: {df['median_holding_period_minutes'].mean():.1f} minutes")
    print(f"   • Max consecutive losses: {df['consecutive_losses_max'].max()}")
    
    # Recommendations
    print(f"\n💡 KEY INSIGHTS:")
    
    if df['model_exit_pct'].mean() > 90:
        print(f"   ⚠️  Model exits dominate ({df['model_exit_pct'].mean():.0f}%) - consider lowering confidence threshold")
    
    if df['avg_holding_period_minutes'].mean() < 5:
        print(f"   ⚠️  Very short holding periods ({df['avg_holding_period_minutes'].mean():.1f}min) - model may be over-reactive")
    
    if df['win_rate'].mean() == 0:
        print(f"   ⚠️  Zero win rate across all tests - signals may not be matching market conditions")
    
    if df['trades_per_day'].mean() > 50:
        print(f"   ⚠️  High trading frequency ({df['trades_per_day'].mean():.0f}/day) - consider higher confidence threshold")
    
    return model_dominant

def analyze_risk_parameters(df):
    """Analyze current risk parameters and suggest optimizations."""
    
    print("\n🛡️  RISK PARAMETER ANALYSIS")
    print("=" * 50)
    
    # Current performance metrics
    print(f"📈 Current Performance:")
    print(f"   • Average return: {df['total_return_pct'].mean():.2f}% ± {df['total_return_pct'].std():.2f}%")
    print(f"   • Best return: {df['total_return_pct'].max():.2f}%")
    print(f"   • Worst return: {df['total_return_pct'].min():.2f}%")
    print(f"   • Win rate: {df['win_rate'].mean():.1%}")
    print(f"   • Max drawdown: {df['max_drawdown_pct'].mean():.2f}%")
    
    # Fee analysis
    print(f"\n💰 Cost Analysis:")
    print(f"   • Average fees per test: ${df['total_fees'].mean():.2f}")
    print(f"   • Fees as % of capital: {df['fees_as_pct_of_capital'].mean():.2f}%")
    print(f"   • Average cost per trade: ${df['total_fees'].sum() / df['total_trades'].sum():.4f}")
    
    # Trading frequency analysis
    print(f"\n📊 Trading Activity:")
    print(f"   • Average trades per day: {df['trades_per_day'].mean():.1f}")
    print(f"   • Total trades executed: {df['total_trades'].sum()}")
    print(f"   • Average execution time: {df['execution_time_seconds'].mean():.1f}s")
    
    # Risk-adjusted returns
    if df['sharpe_ratio'].std() > 0:
        print(f"\n📐 Risk-Adjusted Metrics:")
        print(f"   • Average Sharpe ratio: {df['sharpe_ratio'].mean():.2f}")
        print(f"   • Average Sortino ratio: {df['sortino_ratio'].mean():.2f}")
    
    # Optimization suggestions
    print(f"\n🎯 OPTIMIZATION OPPORTUNITIES:")
    
    if df['total_return_pct'].mean() < -1:
        print(f"   📉 Consistent losses suggest:")
        print(f"      - Lower confidence threshold (currently model exits {df['model_exit_pct'].mean():.0f}% of trades)")
        print(f"      - Longer holding periods (currently {df['avg_holding_period_minutes'].mean():.1f}min average)")
        print(f"      - Different signal filtering approach")
    
    if df['fees_as_pct_of_capital'].mean() > 1:
        print(f"   💸 High trading costs ({df['fees_as_pct_of_capital'].mean():.2f}% of capital):")
        print(f"      - Reduce trading frequency")
        print(f"      - Higher confidence threshold for entries")
        print(f"      - Larger position sizes with fewer trades")
    
    if df['max_drawdown_pct'].max() > 2:
        print(f"   ⚠️  High maximum drawdown ({df['max_drawdown_pct'].max():.2f}%):")
        print(f"      - Consider tighter stop losses")
        print(f"      - Lower position sizes")
        print(f"      - Better entry timing")

def compare_market_conditions(df):
    """Compare performance across different market periods."""
    
    print("\n🌍 MARKET CONDITIONS ANALYSIS")
    print("=" * 50)
    
    # Group by time periods
    df['month'] = pd.to_datetime(df['period_start'], errors='coerce').dt.to_period('M')
    df['period_length_category'] = pd.cut(df['period_days'], 
                                        bins=[0, 2, 7, 30], 
                                        labels=['Short (1-2d)', 'Medium (3-7d)', 'Long (8-30d)'])
    
    # Analysis by month
    if df['month'].nunique() > 1:
        print(f"📅 Performance by Month:")
        monthly_stats = df.groupby('month').agg({
            'total_return_pct': ['mean', 'count'],
            'total_trades': 'sum',
            'win_rate': 'mean'
        }).round(2)
        
        for month in monthly_stats.index:
            returns = monthly_stats.loc[month, ('total_return_pct', 'mean')]
            tests = monthly_stats.loc[month, ('total_return_pct', 'count')]
            trades = monthly_stats.loc[month, ('total_trades', 'sum')]
            win_rate = monthly_stats.loc[month, ('win_rate', 'mean')]
            print(f"   • {month}: {returns:+.2f}% avg return ({tests} tests, {trades} trades, {win_rate:.1%} win rate)")
    
    # Analysis by period length
    print(f"\n⏱️  Performance by Period Length:")
    length_stats = df.groupby('period_length_category').agg({
        'total_return_pct': ['mean', 'count'],
        'trades_per_day': 'mean',
        'win_rate': 'mean'
    }).round(2)
    
    for length in length_stats.index:
        if pd.isna(length):
            continue
        returns = length_stats.loc[length, ('total_return_pct', 'mean')]
        tests = length_stats.loc[length, ('total_return_pct', 'count')]
        trades_per_day = length_stats.loc[length, ('trades_per_day', 'mean')]
        win_rate = length_stats.loc[length, ('win_rate', 'mean')]
        print(f"   • {length}: {returns:+.2f}% avg return ({tests} tests, {trades_per_day:.1f} trades/day, {win_rate:.1%} win rate)")
    
    # Best and worst performing periods
    print(f"\n🏆 Best Performing Period:")
    best_idx = df['total_return_pct'].idxmax()
    best = df.loc[best_idx]
    print(f"   • Period: {best['period_start']} to {best['period_end']}")
    print(f"   • Return: {best['total_return_pct']:+.2f}%")
    print(f"   • Trades: {best['total_trades']} ({best['win_rate']:.1%} win rate)")
    print(f"   • Holding time: {best['avg_holding_period_minutes']:.1f} min avg")
    
    print(f"\n📉 Worst Performing Period:")
    worst_idx = df['total_return_pct'].idxmin()
    worst = df.loc[worst_idx]
    print(f"   • Period: {worst['period_start']} to {worst['period_end']}")
    print(f"   • Return: {worst['total_return_pct']:+.2f}%")
    print(f"   • Trades: {worst['total_trades']} ({worst['win_rate']:.1%} win rate)")
    print(f"   • Holding time: {worst['avg_holding_period_minutes']:.1f} min avg")

def generate_optimization_recommendations(df):
    """Generate specific optimization recommendations based on analysis."""
    
    print("\n🎯 SPECIFIC OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = []
    
    # Confidence threshold optimization
    if df['model_exit_pct'].mean() > 95:
        recommendations.append({
            'category': 'Signal Processing',
            'issue': f"Model exits {df['model_exit_pct'].mean():.0f}% of all trades",
            'recommendation': 'Lower confidence threshold from 0.5 to 0.3',
            'expected_impact': 'More trades held longer, potential for positive returns'
        })
    
    # Holding period optimization
    if df['avg_holding_period_minutes'].mean() < 3:
        recommendations.append({
            'category': 'Exit Strategy',
            'issue': f"Very short holding periods ({df['avg_holding_period_minutes'].mean():.1f} min avg)",
            'recommendation': 'Add minimum holding period of 5-10 minutes',
            'expected_impact': 'Reduce transaction costs, allow patterns to develop'
        })
    
    # Fee optimization
    if df['fees_as_pct_of_capital'].mean() > 1.5:
        recommendations.append({
            'category': 'Cost Management',
            'issue': f"High trading costs ({df['fees_as_pct_of_capital'].mean():.2f}% of capital)",
            'recommendation': 'Increase position sizes and reduce trade frequency',
            'expected_impact': 'Lower relative transaction costs'
        })
    
    # Win rate optimization
    if df['win_rate'].mean() == 0:
        recommendations.append({
            'category': 'Strategy Effectiveness',
            'issue': '0% win rate across all tested periods',
            'recommendation': 'Test different market conditions or retrain model',
            'expected_impact': 'Identify periods where strategy works'
        })
    
    # Trading frequency optimization
    if df['trades_per_day'].mean() > 30:
        recommendations.append({
            'category': 'Trading Frequency',
            'issue': f"High frequency trading ({df['trades_per_day'].mean():.0f} trades/day)",
            'recommendation': 'Increase confidence threshold to 0.7-0.8',
            'expected_impact': 'Focus on highest confidence signals only'
        })
    
    # Period selection optimization
    if len(df) > 3:
        best_periods = df.nlargest(2, 'total_return_pct')
        if best_periods['total_return_pct'].min() > -0.5:
            recommendations.append({
                'category': 'Period Selection',
                'issue': 'Some periods show better performance than others',
                'recommendation': f"Focus testing on periods similar to {best_periods['period_start'].iloc[0]}",
                'expected_impact': 'Better understanding of favorable market conditions'
            })
    
    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['category']}:")
            print(f"   📋 Issue: {rec['issue']}")
            print(f"   💡 Recommendation: {rec['recommendation']}")
            print(f"   🎯 Expected Impact: {rec['expected_impact']}")
    else:
        print("\n✅ No major optimization opportunities identified based on current data.")
    
    return recommendations

def save_analysis_report(df, recommendations, output_dir):
    """Save comprehensive analysis report."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed data
    data_file = output_path / 'backtest_analysis_data.csv'
    df.to_csv(data_file, index=False)
    
    # Generate comprehensive report
    report_file = output_path / 'backtest_optimization_report.md'
    
    with open(report_file, 'w') as f:
        f.write("# Comprehensive Backtest Analysis & Optimization Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Analysis Period:** {len(df)} backtests analyzed\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Tests Analyzed:** {len(df)}\n")
        f.write(f"- **Average Return:** {df['total_return_pct'].mean():.2f}%\n")
        f.write(f"- **Total Trades Executed:** {df['total_trades'].sum()}\n")
        f.write(f"- **Average Win Rate:** {df['win_rate'].mean():.1%}\n")
        f.write(f"- **Model Exit Dominance:** {df['model_exit_pct'].mean():.0f}%\n\n")
        
        # Key Findings
        f.write("## Key Findings\n\n")
        f.write("### Performance Metrics\n")
        f.write(f"- Best performing period: {df.loc[df['total_return_pct'].idxmax(), 'period_start']} ({df['total_return_pct'].max():.2f}%)\n")
        f.write(f"- Worst performing period: {df.loc[df['total_return_pct'].idxmin(), 'period_start']} ({df['total_return_pct'].min():.2f}%)\n")
        f.write(f"- Average holding period: {df['avg_holding_period_minutes'].mean():.1f} minutes\n")
        f.write(f"- Trading frequency: {df['trades_per_day'].mean():.1f} trades per day\n\n")
        
        f.write("### Exit Strategy Analysis\n")
        f.write(f"- Model exits: {df['model_exit_pct'].mean():.1f}% of all trades\n")
        f.write(f"- Stop loss exits: {df['stop_loss_pct'].mean():.1f}% of all trades\n")
        f.write(f"- Take profit exits: {df['take_profit_pct'].mean():.1f}% of all trades\n\n")
        
        # Optimization Recommendations
        f.write("## Optimization Recommendations\n\n")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                f.write(f"### {i}. {rec['category']}\n")
                f.write(f"**Issue:** {rec['issue']}\n\n")
                f.write(f"**Recommendation:** {rec['recommendation']}\n\n")
                f.write(f"**Expected Impact:** {rec['expected_impact']}\n\n")
        
        # Detailed Period Analysis
        f.write("## Detailed Period Analysis\n\n")
        f.write("| Period Start | Period End | Return % | Trades | Win Rate | Avg Hold (min) | Model Exits % |\n")
        f.write("|--------------|------------|----------|--------|----------|----------------|---------------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['period_start']} | {row['period_end']} | {row['total_return_pct']:.2f}% | "
                   f"{row['total_trades']} | {row['win_rate']:.1%} | {row['avg_holding_period_minutes']:.1f} | "
                   f"{row['model_exit_pct']:.0f}% |\n")
    
    print(f"\n📊 Analysis complete!")
    print(f"📂 Data saved to: {data_file}")
    print(f"📋 Report saved to: {report_file}")

def main():
    """Main analysis function."""
    print("📊 Starting Comprehensive Backtest Analysis")
    print("=" * 50)
    
    # Load all results
    results = load_all_backtest_results()
    
    if not results:
        print("❌ No backtest results found to analyze")
        return
    
    # Extract analysis data
    df = extract_analysis_data(results)
    print(f"\n✅ Analyzed {len(df)} backtest results")
    
    # Run comprehensive analysis
    analyze_exit_timing_patterns(df)
    analyze_risk_parameters(df)
    compare_market_conditions(df)
    recommendations = generate_optimization_recommendations(df)
    
    # Save analysis
    output_dir = '/Users/danielwestgaard/progRel/ml-pred/results/backtest_analysis'
    save_analysis_report(df, recommendations, output_dir)
    
    return df, recommendations

if __name__ == "__main__":
    df, recommendations = main()