#!/usr/bin/env python3
"""
Feature Combination Testing Script
Systematically tests different feature combinations to find optimal set.
"""

import os
import yaml
import subprocess
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

class FeatureCombinationTester:
    def __init__(self):
        self.base_config_path = "hypothesis/spread_compression_patterns/config_process_data_v2.yml"
        self.backtest_config_path = "hypothesis/spread_compression_patterns/config_backtest.yml"
        self.results = []
        
        # Feature groups based on consensus scores
        self.feature_groups = {
            'top_tier': [
                'microstructure_momentum_fusion',  # 0.999
                'depth_stability',                 # 0.967
                'volume_ratio',                    # 0.912
                'institutional_footprint',         # 0.878
                'top_predictor_interaction'        # 0.905
            ],
            'high_tier': [
                'cumulative_order_flow',           # 0.637
                'volume_profile_enhanced',         # 0.600
                'ultra_momentum_suite',            # 0.488
                'compression_prediction_strength', # 0.423
                'signal_conviction'                # 0.410
            ],
            'medium_tier': [
                'tick_intensity',                  # 0.367
                'intrabar_pressure',              # 0.365
                'microstructure_health',          # 0.337
                'liquidity_stress',               # 0.331
                'depth_variability'               # 0.287
            ],
            'context_tier': [
                'london_session',                  # 0.175
                'price_level_support',            # 0.244
                'spread_regime_stability'         # 0.159
            ]
        }
    
    def create_test_combinations(self):
        """Create systematic test combinations"""
        combinations = []
        
        # Test 1: Core combinations (top tier only, different sizes)
        top_features = self.feature_groups['top_tier']
        for i in range(3, len(top_features) + 1):
            from itertools import combinations as iter_combinations
            for combo in iter_combinations(top_features, i):
                combinations.append({
                    'name': f'top_tier_{i}feat',
                    'features': list(combo),
                    'description': f'Top tier features only ({i} features)'
                })
        
        # Test 2: Enhanced-only vs Traditional-only
        enhanced_features = ['microstructure_momentum_fusion', 'top_predictor_interaction', 
                           'volume_profile_enhanced', 'ultra_momentum_suite']
        traditional_features = ['depth_stability', 'volume_ratio', 'institutional_footprint', 
                              'cumulative_order_flow', 'compression_prediction_strength']
        
        combinations.extend([
            {
                'name': 'enhanced_only',
                'features': enhanced_features,
                'description': 'Enhanced features only'
            },
            {
                'name': 'traditional_only', 
                'features': traditional_features,
                'description': 'Traditional features only'
            }
        ])
        
        # Test 3: Incremental addition (start with top 3, add one by one)
        base_features = ['microstructure_momentum_fusion', 'depth_stability', 'volume_ratio']
        additional_features = [
            'institutional_footprint',
            'top_predictor_interaction', 
            'cumulative_order_flow',
            'volume_profile_enhanced',
            'compression_prediction_strength',
            'signal_conviction',
            'london_session'
        ]
        
        current_features = base_features.copy()
        for i, feature in enumerate(additional_features):
            current_features.append(feature)
            combinations.append({
                'name': f'incremental_{len(current_features)}feat',
                'features': current_features.copy(),
                'description': f'Incremental build - {len(current_features)} features'
            })
        
        # Test 4: Specialized combinations
        combinations.extend([
            {
                'name': 'volume_focused',
                'features': ['volume_ratio', 'volume_profile_enhanced', 'institutional_footprint', 
                           'cumulative_order_flow', 'microstructure_momentum_fusion'],
                'description': 'Volume-focused combination'
            },
            {
                'name': 'depth_focused',
                'features': ['depth_stability', 'depth_variability', 'liquidity_stress',
                           'microstructure_health', 'top_predictor_interaction'],
                'description': 'Market depth focused combination'  
            },
            {
                'name': 'signal_focused',
                'features': ['compression_prediction_strength', 'signal_conviction', 
                           'microstructure_momentum_fusion', 'top_predictor_interaction'],
                'description': 'Signal strength focused combination'
            }
        ])
        
        return combinations
    
    def update_config_file(self, features):
        """Update the config file with new feature set"""
        with open(self.base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['data_pipeline']['selected_features'] = features
        
        with open(self.base_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def run_training_and_backtest(self, combination):
        """Run training and backtesting for a feature combination"""
        print(f"\n🧪 Testing: {combination['name']} - {combination['description']}")
        print(f"Features ({len(combination['features'])}): {combination['features']}")
        
        try:
            # Update config with new features
            self.update_config_file(combination['features'])
            
            # Run data processing and training
            print("📊 Running data processing and training...")
            result = subprocess.run([
                'python', 'main.py', '--hyp', self.base_config_path
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print(f"❌ Training failed: {result.stderr}")
                return None
            
            # Extract model path from logs
            model_path = self.extract_model_path(result.stdout)
            if not model_path:
                print("❌ Could not find model path")
                return None
                
            # Run backtest
            print("🎯 Running backtest...")
            backtest_result = subprocess.run([
                'python', 'main.py', '--hyp', self.backtest_config_path
            ], capture_output=True, text=True, timeout=600)
            
            if backtest_result.returncode != 0:
                print(f"❌ Backtest failed: {backtest_result.stderr}")
                return None
            
            # Extract performance metrics
            metrics = self.extract_performance_metrics(model_path)
            if metrics:
                metrics.update({
                    'combination_name': combination['name'],
                    'feature_count': len(combination['features']),
                    'features': combination['features'],
                    'description': combination['description']
                })
                print(f"✅ Results: Return: {metrics.get('total_return_pct', 'N/A'):.2f}%, "
                      f"Trades: {metrics.get('total_trades', 'N/A')}, "
                      f"Win Rate: {metrics.get('win_rate', 'N/A'):.1%}")
                return metrics
            
        except subprocess.TimeoutExpired:
            print("⏰ Test timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        return None
    
    def extract_model_path(self, stdout):
        """Extract the latest model path from stdout"""
        # Look for model path in the output
        lines = stdout.split('\n')
        for line in lines:
            if 'storage/models/' in line and 'created' in line.lower():
                # Extract path from line
                import re
                match = re.search(r'storage/models/[\d-_:]+', line)
                if match:
                    return match.group(0)
        return None
    
    def extract_performance_metrics(self, model_path):
        """Extract performance metrics from backtest results"""
        try:
            # Find the latest backtest result file
            backtest_dir = f"{model_path}/backtesting"
            if not os.path.exists(backtest_dir):
                return None
                
            result_files = [f for f in os.listdir(backtest_dir) if f.startswith('backtest_results_')]
            if not result_files:
                return None
                
            latest_file = max(result_files)
            result_path = os.path.join(backtest_dir, latest_file)
            
            with open(result_path, 'r') as f:
                data = json.load(f)
            
            return {
                'total_return_pct': data['backtest_summary']['total_return_pct'],
                'total_trades': data['trading_performance']['total_trades'],
                'win_rate': data['trading_performance']['win_rate'],
                'max_drawdown_pct': data['trading_performance']['max_drawdown_pct'],
                'sharpe_ratio': data['risk_metrics']['sharpe_ratio'],
                'profit_factor': data['risk_metrics']['profit_factor']
            }
            
        except Exception as e:
            print(f"Warning: Could not extract metrics - {e}")
            return None
    
    def run_all_tests(self, max_tests=None):
        """Run all feature combination tests"""
        combinations = self.create_test_combinations()
        
        if max_tests:
            combinations = combinations[:max_tests]
        
        print(f"🚀 Starting feature combination testing with {len(combinations)} combinations")
        
        for i, combination in enumerate(combinations, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}/{len(combinations)}")
            
            result = self.run_training_and_backtest(combination)
            if result:
                self.results.append(result)
        
        # Save results
        self.save_results()
        self.analyze_results()
    
    def save_results(self):
        """Save test results to CSV"""
        if not self.results:
            print("No results to save")
            return
            
        df = pd.DataFrame(self.results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"feature_combination_results_{timestamp}.csv"
        df.to_csv(results_path, index=False)
        print(f"\n💾 Results saved to: {results_path}")
    
    def analyze_results(self):
        """Analyze and rank results"""
        if not self.results:
            print("No results to analyze")
            return
            
        df = pd.DataFrame(self.results)
        
        # Rank by multiple criteria
        df['performance_score'] = (
            df['total_return_pct'] * 0.4 +           # 40% weight on returns
            (1 - df['max_drawdown_pct']) * 0.3 +     # 30% weight on drawdown (inverted)
            df['win_rate'] * 100 * 0.2 +             # 20% weight on win rate  
            df['profit_factor'] * 0.1                # 10% weight on profit factor
        )
        
        df_sorted = df.sort_values('performance_score', ascending=False)
        
        print(f"\n🏆 TOP 5 FEATURE COMBINATIONS:")
        print("="*80)
        for i, (_, row) in enumerate(df_sorted.head().iterrows(), 1):
            print(f"{i}. {row['combination_name']} ({row['feature_count']} features)")
            print(f"   Return: {row['total_return_pct']:.2f}% | Drawdown: {row['max_drawdown_pct']:.2f}% | "
                  f"Win Rate: {row['win_rate']:.1%} | Trades: {row['total_trades']}")
            print(f"   Features: {row['features']}")
            print(f"   Score: {row['performance_score']:.2f}")
            print()

if __name__ == "__main__":
    tester = FeatureCombinationTester()
    
    # Run limited test first (top 10 combinations)
    print("Running feature combination tests...")
    tester.run_all_tests(max_tests=10)