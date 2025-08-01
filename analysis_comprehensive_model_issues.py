#!/usr/bin/env python3
"""
Comprehensive Analysis: Model Performance Issues
Identifies root causes keeping the model from being truly predictive
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelPerformanceAnalyzer:
    def __init__(self):
        self.results = {}
        
    def analyze_backtest_results(self, backtest_log_path):
        """Analyze backtest performance patterns"""
        print("=== BACKTEST PERFORMANCE ANALYSIS ===")
        
        # Read the latest backtest results JSON
        model_folder = Path("storage/models/2025-07-31_22:57:51")
        backtest_results_path = model_folder / "backtesting/backtest_results_20250801_103142.json"
        
        if backtest_results_path.exists():
            with open(backtest_results_path, 'r') as f:
                backtest_data = json.load(f)
                
            print(f"Total Trades: {backtest_data.get('total_trades', 'N/A')}")
            print(f"Win Rate: {backtest_data.get('win_rate', 'N/A')}%")
            print(f"Total Return: {backtest_data.get('total_return_pct', 'N/A')}%")
            print(f"Sharpe Ratio: {backtest_data.get('sharpe_ratio', 'N/A')}")
            
            # Analyze individual trades
            trades = backtest_data.get('trades', [])
            if trades:
                pnls = [trade.get('pnl', 0) for trade in trades]
                print(f"Average P&L per trade: ${np.mean(pnls):.4f}")
                print(f"Median P&L per trade: ${np.median(pnls):.4f}")
                print(f"Winning trades: {sum(1 for pnl in pnls if pnl > 0)}")
                print(f"Losing trades: {sum(1 for pnl in pnls if pnl < 0)}")
                
                # Analyze trade directions
                buy_trades = [t for t in trades if t.get('direction') == 'BUY']
                sell_trades = [t for t in trades if t.get('direction') == 'SELL']
                print(f"BUY trades: {len(buy_trades)}, Average P&L: ${np.mean([t.get('pnl', 0) for t in buy_trades]):.4f}")
                print(f"SELL trades: {len(sell_trades)}, Average P&L: ${np.mean([t.get('pnl', 0) for t in sell_trades]):.4f}")
                
            self.results['backtest'] = {
                'total_trades': backtest_data.get('total_trades', 0),
                'win_rate': backtest_data.get('win_rate', 0),
                'avg_pnl': np.mean(pnls) if trades else 0,
                'profitable_trades': sum(1 for pnl in pnls if pnl > 0) if trades else 0
            }
        
    def analyze_feature_importance(self):
        """Analyze feature importance and predictive power"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Read consensus ranking
        fi_path = Path("storage/data_engineering_stats/feature_importance/2025-07-31_22-48-05/consensus_ranking.csv")
        if fi_path.exists():
            fi_df = pd.read_csv(fi_path, index_col=0)
            print("Top features by consensus score:")
            for idx, (feature, score) in enumerate(fi_df.iterrows(), 1):
                print(f"{idx}. {feature}: {score['consensus_score']:.4f}")
                
            # Check if scores are meaningful
            max_score = fi_df['consensus_score'].max()
            min_score = fi_df['consensus_score'].min()
            score_range = max_score - min_score
            
            print(f"\nFeature importance range: {min_score:.4f} - {max_score:.4f} (spread: {score_range:.4f})")
            
            if score_range < 0.5:
                print("⚠️ WARNING: Low feature importance spread suggests weak predictive signals")
                
            self.results['feature_importance'] = {
                'max_score': max_score,
                'min_score': min_score,
                'score_range': score_range,
                'top_feature': fi_df.index[0] if len(fi_df) > 0 else None
            }
    
    def analyze_target_distribution(self):
        """Analyze target distribution and balance"""
        print("\n=== TARGET DISTRIBUTION ANALYSIS ===")
        
        # Based on the log data we saw:
        # Class 0 (HOLD): 2528515 (87.5%)
        # Class 1 (BUY):  194456 (6.7%)  
        # Class 2 (SELL): 167259 (5.8%)
        
        total_samples = 2528515 + 194456 + 167259
        hold_pct = 2528515 / total_samples * 100
        buy_pct = 194456 / total_samples * 100
        sell_pct = 167259 / total_samples * 100
        
        print(f"Target Distribution:")
        print(f"  HOLD: {hold_pct:.1f}%")
        print(f"  BUY:  {buy_pct:.1f}%")
        print(f"  SELL: {sell_pct:.1f}%")
        
        signal_rate = buy_pct + sell_pct
        print(f"Total signal rate: {signal_rate:.1f}%")
        
        # Balance analysis
        balance_ratio = min(buy_pct, sell_pct) / max(buy_pct, sell_pct)
        print(f"BUY/SELL balance ratio: {balance_ratio:.3f} (1.0 = perfect)")
        
        if signal_rate < 15:
            print("⚠️ WARNING: Signal rate below 15% may indicate overly restrictive target criteria")
        if balance_ratio < 0.7:
            print("⚠️ WARNING: Significant imbalance between BUY/SELL signals")
            
        self.results['target_distribution'] = {
            'signal_rate': signal_rate,
            'balance_ratio': balance_ratio,
            'hold_dominance': hold_pct > 85
        }
    
    def identify_root_causes(self):
        """Identify potential root causes of poor performance"""
        print("\n=== ROOT CAUSE ANALYSIS ===")
        
        issues = []
        
        # Issue 1: Poor win rate with many trades
        if self.results.get('backtest', {}).get('win_rate', 0) == 0:
            issues.append({
                'issue': 'Zero Win Rate',
                'severity': 'CRITICAL',
                'description': 'Model predictions are not directionally accurate',
                'potential_causes': [
                    'Target generation logic creates non-predictive labels',
                    'Features lack directional/temporal information',
                    'Data leakage in reverse direction (model learns wrong patterns)',
                    'Insufficient lookahead in target definition'
                ]
            })
        
        # Issue 2: Low feature importance spread  
        fi_range = self.results.get('feature_importance', {}).get('score_range', 0)
        if fi_range < 0.5:
            issues.append({
                'issue': 'Weak Feature Signals',
                'severity': 'HIGH',
                'description': 'Features have low predictive power',
                'potential_causes': [
                    'Missing key market microstructure signals',
                    'Lack of temporal/sequential features',
                    'Missing cross-asset correlations',
                    'Insufficient market regime detection'
                ]
            })
        
        # Issue 3: Target distribution issues
        target_info = self.results.get('target_distribution', {})
        if target_info.get('signal_rate', 0) < 15:
            issues.append({
                'issue': 'Low Signal Rate',
                'severity': 'MEDIUM', 
                'description': 'Too few trading opportunities generated',
                'potential_causes': [
                    'Overly restrictive target criteria',
                    'Market conditions not captured well',
                    'Thresholds not calibrated to market volatility'
                ]
            })
        
        # Issue 4: Consistent small losses
        avg_pnl = self.results.get('backtest', {}).get('avg_pnl', 0)
        if avg_pnl < -0.1:  # Consistent small losses
            issues.append({
                'issue': 'Systematic Negative Bias',
                'severity': 'CRITICAL',
                'description': 'Model consistently predicts wrong direction',
                'potential_causes': [
                    'Target labels inverted or incorrect',
                    'Look-ahead bias in target generation',
                    'Model learns transaction costs instead of alpha',
                    'Features capture lagging rather than leading indicators'
                ]
            })
        
        # Print issues
        for i, issue in enumerate(issues, 1):
            print(f"\n{i}. {issue['issue']} ({issue['severity']})")
            print(f"   Description: {issue['description']}")
            print(f"   Potential causes:")
            for cause in issue['potential_causes']:
                print(f"     - {cause}")
        
        self.results['root_causes'] = issues
        return issues
    
    def generate_recommendations(self):
        """Generate actionable recommendations"""
        print("\n=== ACTIONABLE RECOMMENDATIONS ===")
        
        recommendations = []
        
        # Critical fixes
        recommendations.extend([
            {
                'priority': 'CRITICAL',
                'category': 'Target Engineering',
                'action': 'Add Forward-Looking Target',
                'description': 'Modify target to predict future price movement (5-15 minutes ahead)',
                'implementation': 'Add future_return = close.shift(-10) / close - 1 to target logic'
            },
            {
                'priority': 'CRITICAL', 
                'category': 'Feature Engineering',
                'action': 'Add Sequential/Temporal Features',
                'description': 'Include features that capture market momentum and trend',
                'implementation': 'Add: price_momentum_5min, volume_trend, volatility_expansion'
            },
            {
                'priority': 'HIGH',
                'category': 'Feature Engineering', 
                'action': 'Add Market Microstructure',
                'description': 'Include bid-ask spread, order book imbalance, tick-level features',
                'implementation': 'Add: spread_compression_rate, book_imbalance, tick_momentum'
            }
        ])
        
        # High priority improvements
        recommendations.extend([
            {
                'priority': 'HIGH',
                'category': 'Model Architecture',
                'action': 'Test Time Series Models',
                'description': 'Try LSTM/GRU or sequence-aware models',
                'implementation': 'Implement CNN-LSTM or Transformer for temporal patterns'
            },
            {
                'priority': 'HIGH',
                'category': 'Target Engineering',
                'action': 'Multi-Horizon Targets',
                'description': 'Create targets for different time horizons (1min, 5min, 15min)',
                'implementation': 'Generate separate targets for short/medium/long-term movements'
            }
        ])
        
        # Medium priority enhancements
        recommendations.extend([
            {
                'priority': 'MEDIUM',
                'category': 'Feature Engineering',
                'action': 'Add Market Regime Detection',
                'description': 'Features that identify market conditions (trending/ranging/volatile)',
                'implementation': 'Add: volatility_regime, trend_strength, market_phase'
            },
            {
                'priority': 'MEDIUM',
                'category': 'Data Quality',
                'action': 'Feature Engineering Pipeline Audit',
                'description': 'Verify all features are calculated correctly without lookahead',
                'implementation': 'Review all .shift() operations and ensure proper time alignment'
            }
        ])
        
        # Print recommendations by priority
        for priority in ['CRITICAL', 'HIGH', 'MEDIUM']:
            priority_recs = [r for r in recommendations if r['priority'] == priority]
            if priority_recs:
                print(f"\n{priority} PRIORITY:")
                for i, rec in enumerate(priority_recs, 1):
                    print(f"{i}. {rec['action']} ({rec['category']})")
                    print(f"   {rec['description']}")
                    print(f"   Implementation: {rec['implementation']}")
        
        return recommendations
    
    def run_complete_analysis(self):
        """Run complete analysis"""
        print("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        self.analyze_backtest_results("logs/backtesting/logging_01082025_103003.log")
        self.analyze_feature_importance() 
        self.analyze_target_distribution()
        
        root_causes = self.identify_root_causes()
        recommendations = self.generate_recommendations()
        
        # Summary
        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"Root causes identified: {len(root_causes)}")
        print(f"Recommendations generated: {len(recommendations)}")
        print(f"Critical issues: {len([r for r in recommendations if r['priority'] == 'CRITICAL'])}")
        
        return {
            'root_causes': root_causes,
            'recommendations': recommendations,
            'results': self.results
        }

if __name__ == "__main__":
    analyzer = ModelPerformanceAnalyzer()
    analysis = analyzer.run_complete_analysis()