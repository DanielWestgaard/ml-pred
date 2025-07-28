#!/usr/bin/env python3
"""
Systematic Backtest Runner
==========================

Run backtests across multiple signal-rich periods to identify:
1. Optimal risk parameters
2. Performance across different market conditions  
3. Model exit timing vs manual stop loss/take profit effectiveness
"""

import subprocess
import json
import yaml
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import os

def generate_test_periods():
    """Generate systematic test periods across different months and market conditions."""
    
    test_periods = [
        # Recent periods (2025) - likely to have more signals
        {'name': 'Feb_2025_week1', 'start': '2025-02-01', 'end': '2025-02-08', 'description': 'Early February - potential volatility'},
        {'name': 'Feb_2025_week2', 'start': '2025-02-08', 'end': '2025-02-15', 'description': 'Mid February - market stabilization'},
        {'name': 'Feb_2025_week3', 'start': '2025-02-15', 'end': '2025-02-22', 'description': 'Late February - end month effects'},
        {'name': 'Mar_2025_week1', 'start': '2025-03-01', 'end': '2025-03-08', 'description': 'Early March - new month momentum'},
        {'name': 'Mar_2025_week2', 'start': '2025-03-08', 'end': '2025-03-15', 'description': 'Mid March - steady trading'},
        {'name': 'Apr_2025_week1', 'start': '2025-04-01', 'end': '2025-04-08', 'description': 'Early April - Q2 start'},
        {'name': 'Apr_2025_week2', 'start': '2025-04-08', 'end': '2025-04-15', 'description': 'Mid April - spring trading'},
        {'name': 'May_2025_week1', 'start': '2025-05-01', 'end': '2025-05-08', 'description': 'Early May - spring volatility'},
        
        # 2024 periods for comparison
        {'name': 'Nov_2024_week1', 'start': '2024-11-01', 'end': '2024-11-08', 'description': 'Post-election period'},
        {'name': 'Nov_2024_week2', 'start': '2024-11-08', 'end': '2024-11-15', 'description': 'Mid November - market reaction'},
        {'name': 'Dec_2024_week1', 'start': '2024-12-01', 'end': '2024-12-08', 'description': 'Early December - year-end positioning'},
        {'name': 'Dec_2024_week2', 'start': '2024-12-08', 'end': '2024-12-15', 'description': 'Mid December - holiday effects'},
        {'name': 'Jan_2025_week1', 'start': '2025-01-01', 'end': '2025-01-08', 'description': 'New Year - fresh positioning'},
        {'name': 'Jan_2025_week2', 'start': '2025-01-08', 'end': '2025-01-15', 'description': 'Mid January - normal trading resumes'},
        
        # Known active periods based on documentation
        {'name': 'Known_Active_Feb3', 'start': '2025-02-03', 'end': '2025-02-04', 'description': 'Known signal-rich period from docs'},
    ]
    
    return test_periods

def generate_backtest_config(period, base_config_path, risk_params=None):
    """Generate backtest config for a specific period."""
    
    # Load base configuration
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update period
    config['backtesting']['start_date'] = period['start']
    config['backtesting']['end_date'] = period['end']
    
    # Adjust max_loops for longer periods
    days = (datetime.strptime(period['end'], '%Y-%m-%d') - 
           datetime.strptime(period['start'], '%Y-%m-%d')).days
    config['backtesting']['execution']['max_loops'] = max(2000, days * 200)  # ~200 loops per day
    
    # Apply risk parameters if provided
    if risk_params:
        for key, value in risk_params.items():
            if 'risk_management' in config['backtesting']:
                config['backtesting']['risk_management'][key] = value
    
    return config

def run_backtest(config, period_name, output_dir):
    """Run a single backtest and return results."""
    
    # Create temporary config file
    config_path = Path(output_dir) / f"temp_config_{period_name}.yml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"🚀 Running backtest for {period_name} ({config['backtesting']['start_date']} to {config['backtesting']['end_date']})")
    
    try:
        # Run backtest
        start_time = time.time()
        result = subprocess.run([
            'python', 'main.py', '-hyp', str(config_path)
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"   ✅ Completed in {execution_time:.1f}s")
            
            # Try to find the results file
            model_dir = Path(config['backtesting']['model_folder'])
            backtest_dir = model_dir / 'backtesting'
            
            # Find the most recent results file
            if backtest_dir.exists():
                result_files = list(backtest_dir.glob('backtest_results_*.json'))
                if result_files:
                    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                    
                    with open(latest_file, 'r') as f:
                        backtest_results = json.load(f)
                    
                    # Add metadata
                    backtest_results['period_info'] = {
                        'name': period_name,
                        'start_date': config['backtesting']['start_date'],
                        'end_date': config['backtesting']['end_date'],
                        'execution_time': execution_time,
                        'config_used': config['backtesting']
                    }
                    
                    return backtest_results
            
            print(f"   ⚠️  Results file not found for {period_name}")
            return None
            
        else:
            print(f"   ❌ Failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"   ⏰ Timeout after 5 minutes")
        return None
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None
    finally:
        # Clean up temp config
        if config_path.exists():
            config_path.unlink()

def run_risk_parameter_tests(base_periods, base_config_path, output_dir):
    """Test different risk parameter combinations."""
    
    risk_variants = [
        {'name': 'conservative', 'params': {'stop_loss_pct': 0.01, 'take_profit_pct': 0.02, 'max_position_size': 0.01}},
        {'name': 'moderate', 'params': {'stop_loss_pct': 0.02, 'take_profit_pct': 0.04, 'max_position_size': 0.02}},  # Default
        {'name': 'aggressive', 'params': {'stop_loss_pct': 0.03, 'take_profit_pct': 0.06, 'max_position_size': 0.03}},
        {'name': 'tight_stops', 'params': {'stop_loss_pct': 0.005, 'take_profit_pct': 0.02, 'max_position_size': 0.02}},
        {'name': 'wide_targets', 'params': {'stop_loss_pct': 0.02, 'take_profit_pct': 0.08, 'max_position_size': 0.02}},
    ]
    
    results = []
    
    # Test on a few representative periods
    test_periods = base_periods[:3]  # First 3 periods
    
    for risk_variant in risk_variants:
        print(f"\n🔧 Testing {risk_variant['name']} risk parameters...")
        
        for period in test_periods:
            config = generate_backtest_config(period, base_config_path, risk_variant['params'])
            period_name = f"{period['name']}_{risk_variant['name']}"
            
            result = run_backtest(config, period_name, output_dir)
            if result:
                result['risk_variant'] = risk_variant['name']
                result['risk_params'] = risk_variant['params']
                results.append(result)
    
    return results

def analyze_results(all_results, output_dir):
    """Analyze all backtest results and generate insights."""
    
    if not all_results:
        print("❌ No results to analyze")
        return
    
    # Convert to DataFrame for analysis
    analysis_data = []
    
    for result in all_results:
        period_info = result.get('period_info', {})
        trading_perf = result.get('trading_performance', {})
        risk_metrics = result.get('risk_metrics', {})
        trade_analysis = result.get('trade_analysis', {})
        backtest_summary = result.get('backtest_summary', {})
        
        analysis_data.append({
            'period_name': period_info.get('name', 'unknown'),
            'start_date': period_info.get('start_date', ''),
            'end_date': period_info.get('end_date', ''),
            'execution_time': period_info.get('execution_time', 0),
            'total_return_pct': backtest_summary.get('total_return_pct', 0),
            'total_trades': trading_perf.get('total_trades', 0),
            'win_rate': trading_perf.get('win_rate', 0),
            'max_drawdown_pct': trading_perf.get('max_drawdown_pct', 0),
            'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
            'profit_factor': risk_metrics.get('profit_factor', 0),
            'avg_holding_period': trade_analysis.get('average_holding_period_minutes', 0),
            'risk_variant': result.get('risk_variant', 'default'),
            'avg_pnl_per_trade': trade_analysis.get('average_pnl_per_trade', 0),
        })
    
    df = pd.DataFrame(analysis_data)
    
    # Save detailed results
    results_file = Path(output_dir) / 'systematic_backtest_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate analysis report
    report_file = Path(output_dir) / 'backtest_analysis_report.md'
    
    with open(report_file, 'w') as f:
        f.write("# Systematic Backtest Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Tests:** {len(all_results)}\n\n")
        
        # Performance Summary
        f.write("## Performance Summary\n\n")
        f.write("### Top Performing Periods\n")
        top_performers = df.nlargest(5, 'total_return_pct')
        f.write("| Period | Return % | Trades | Win Rate | Sharpe | Max DD % |\n")
        f.write("|--------|----------|--------|----------|--------|-----------|\n")
        for _, row in top_performers.iterrows():
            f.write(f"| {row['period_name']} | {row['total_return_pct']:.2f}% | {row['total_trades']} | "
                   f"{row['win_rate']:.1%} | {row['sharpe_ratio']:.2f} | {row['max_drawdown_pct']:.2f}% |\n")
        
        f.write("\n### Performance Statistics\n")
        f.write(f"- **Average Return**: {df['total_return_pct'].mean():.2f}%\n")
        f.write(f"- **Best Return**: {df['total_return_pct'].max():.2f}% ({df.loc[df['total_return_pct'].idxmax(), 'period_name']})\n")
        f.write(f"- **Worst Return**: {df['total_return_pct'].min():.2f}% ({df.loc[df['total_return_pct'].idxmin(), 'period_name']})\n")
        f.write(f"- **Average Trades per Period**: {df['total_trades'].mean():.1f}\n")
        f.write(f"- **Average Win Rate**: {df['win_rate'].mean():.1%}\n")
        f.write(f"- **Average Holding Period**: {df['avg_holding_period'].mean():.1f} minutes\n")
        
        # Risk Parameter Analysis
        if 'risk_variant' in df.columns and df['risk_variant'].nunique() > 1:
            f.write("\n## Risk Parameter Analysis\n\n")
            risk_summary = df.groupby('risk_variant').agg({
                'total_return_pct': ['mean', 'std'],
                'win_rate': 'mean',
                'max_drawdown_pct': 'mean',
                'sharpe_ratio': 'mean'
            }).round(3)
            
            f.write("| Risk Strategy | Avg Return % | Return Std % | Avg Win Rate | Avg Max DD % | Avg Sharpe |\n")
            f.write("|---------------|--------------|--------------|--------------|--------------|------------|\n")
            for variant in risk_summary.index:
                row = risk_summary.loc[variant]
                f.write(f"| {variant} | {row[('total_return_pct', 'mean')]:.2f} | "
                       f"{row[('total_return_pct', 'std')]:.2f} | {row[('win_rate', 'mean')]:.1%} | "
                       f"{row[('max_drawdown_pct', 'mean')]:.2f} | {row[('sharpe_ratio', 'mean')]:.2f} |\n")
        
        # Market Conditions Analysis
        f.write("\n## Period Analysis\n\n")
        f.write("### All Periods Tested\n")
        f.write("| Period | Start | End | Return % | Trades | Win Rate | Execution Time |\n")
        f.write("|--------|-------|-----|----------|--------|----------|----------------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['period_name']} | {row['start_date']} | {row['end_date']} | "
                   f"{row['total_return_pct']:.2f}% | {row['total_trades']} | "
                   f"{row['win_rate']:.1%} | {row['execution_time']:.1f}s |\n")
    
    print(f"\n📊 Analysis complete!")
    print(f"📂 Results saved to: {results_file}")
    print(f"📋 Report saved to: {report_file}")
    
    # Print summary to console
    print(f"\n🏆 BACKTEST SUMMARY:")
    print(f"   • Total tests: {len(all_results)}")
    print(f"   • Best period: {df.loc[df['total_return_pct'].idxmax(), 'period_name']} ({df['total_return_pct'].max():.2f}%)")
    print(f"   • Worst period: {df.loc[df['total_return_pct'].idxmin(), 'period_name']} ({df['total_return_pct'].min():.2f}%)")
    print(f"   • Average return: {df['total_return_pct'].mean():.2f}%")
    print(f"   • Periods with positive returns: {(df['total_return_pct'] > 0).sum()}/{len(df)}")
    
    return df

def main():
    """Main execution function."""
    print("🚀 Starting Systematic Backtest Runner")
    print("=" * 50)
    
    # Setup
    base_config_path = '/Users/danielwestgaard/progRel/ml-pred/hypothesis/spread_compression_patterns/config_backtest.yml'
    output_dir = Path('/Users/danielwestgaard/progRel/ml-pred/results/systematic_backtests')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test periods
    test_periods = generate_test_periods()
    print(f"📅 Generated {len(test_periods)} test periods")
    
    all_results = []
    
    # Run basic period tests
    print(f"\n🔍 Running backtests on {len(test_periods)} periods...")
    for i, period in enumerate(test_periods):
        print(f"\n[{i+1}/{len(test_periods)}] Testing: {period['name']} - {period['description']}")
        
        config = generate_backtest_config(period, base_config_path)
        result = run_backtest(config, period['name'], output_dir)
        
        if result:
            all_results.append(result)
            # Quick feedback on result
            perf = result.get('trading_performance', {})
            summary = result.get('backtest_summary', {})
            print(f"   📈 Result: {summary.get('total_return_pct', 0):.2f}% return, "
                  f"{perf.get('total_trades', 0)} trades, {perf.get('win_rate', 0):.1%} win rate")
        
        time.sleep(1)  # Brief pause between tests
    
    # Run risk parameter optimization tests
    print(f"\n🔧 Running risk parameter optimization tests...")
    risk_results = run_risk_parameter_tests(test_periods[:3], base_config_path, output_dir)
    all_results.extend(risk_results)
    
    # Analyze all results
    print(f"\n📊 Analyzing {len(all_results)} total results...")
    analysis_df = analyze_results(all_results, output_dir)
    
    print(f"\n✅ Systematic backtesting complete!")
    print(f"🎯 Key insights:")
    if len(all_results) > 0:
        best_result = max(all_results, key=lambda x: x.get('backtest_summary', {}).get('total_return_pct', -999))
        print(f"   • Best performing period: {best_result['period_info']['name']}")
        print(f"   • Best return: {best_result['backtest_summary']['total_return_pct']:.2f}%")
        print(f"   • Most trades: {max(r.get('trading_performance', {}).get('total_trades', 0) for r in all_results)}")
    
    return all_results, analysis_df

if __name__ == "__main__":
    main()