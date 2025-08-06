"""
Rigorous validation framework to prevent data leakage and lookahead bias
in spread expansion hypothesis testing.

Key principles:
1. Chronological splits only (no random sampling)
2. Parameters calculated ONLY on training data
3. Multiple out-of-sample periods tested
4. Walk-forward validation
5. Bootstrap confidence intervals
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class RigorousValidator:
    """
    Implements multiple validation techniques to ensure hypothesis robustness
    """
    
    def __init__(self, csv_file_path):
        """Initialize with data"""
        print(f"Loading data: {csv_file_path}")
        self.df = pd.read_csv(csv_file_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Calculate derived metrics
        self._prepare_data()
        print(f"Loaded {len(self.df)} records from {self.df['Date'].min()} to {self.df['Date'].max()}")
    
    def _prepare_data(self):
        """Prepare data for analysis"""
        # Mid price and forward returns
        self.df['mid_price'] = (self.df['bid_close'] + self.df['ask_close']) / 2
        
        for minutes in [15, 30, 45, 60]:
            self.df[f'return_{minutes}m'] = (
                self.df['mid_price'].shift(-minutes) - self.df['mid_price']
            ) / self.df['mid_price']
    
    def chronological_validation(self, test_periods=3, min_train_months=6):
        """
        Test hypothesis across multiple chronological out-of-sample periods
        """
        print("="*60)
        print("CHRONOLOGICAL VALIDATION")
        print("="*60)
        
        # Calculate total months of data
        total_days = (self.df['Date'].max() - self.df['Date'].min()).days
        total_months = total_days / 30.44
        
        if total_months < min_train_months + 2:
            print(f"⚠️  Insufficient data: {total_months:.1f} months")
            return pd.DataFrame()
        
        # Create chronological splits
        results = []
        
        for test_period in range(test_periods):
            print(f"\n--- TEST PERIOD {test_period + 1} ---")
            
            # Calculate split points
            test_start_pct = 0.7 + (test_period * 0.1)  # 70%, 80%, 90%
            test_end_pct = min(1.0, test_start_pct + 0.2)  # 20% test periods
            
            train_end_idx = int(len(self.df) * test_start_pct)
            test_start_idx = train_end_idx
            test_end_idx = int(len(self.df) * test_end_pct)
            
            # Split data
            train_df = self.df.iloc[:train_end_idx].copy()
            test_df = self.df.iloc[test_start_idx:test_end_idx].copy()
            
            if len(test_df) < 1000:  # Need minimum test samples
                continue
            
            print(f"Training: {train_df['Date'].min()} to {train_df['Date'].max()}")
            print(f"Testing:  {test_df['Date'].min()} to {test_df['Date'].max()}")
            print(f"Train samples: {len(train_df):,}, Test samples: {len(test_df):,}")
            
            # Calculate thresholds on training data ONLY
            train_spread_90 = train_df['spread_close'].quantile(0.90)
            train_spread_85 = train_df['spread_close'].quantile(0.85)
            train_vol_avg = train_df['Volume'].mean()
            
            # Test each threshold
            for threshold_name, threshold_val in [
                ('90th_pct', train_spread_90),
                ('85th_pct', train_spread_85)
            ]:
                # Apply to test data
                expansion_mask = (
                    (test_df['spread_close'] >= threshold_val) &
                    (test_df['Volume'] > train_vol_avg)
                )
                
                expansion_events = test_df[expansion_mask]
                
                if len(expansion_events) < 20:
                    continue
                
                # Test 45 and 60 minute returns
                for minutes in [45, 60]:
                    return_col = f'return_{minutes}m'
                    
                    if return_col not in test_df.columns:
                        continue
                    
                    exp_returns = expansion_events[return_col].dropna()
                    all_returns = test_df[return_col].dropna()
                    
                    if len(exp_returns) < 10:
                        continue
                    
                    # Statistical tests
                    mean_exp = exp_returns.mean()
                    mean_all = all_returns.mean()
                    
                    try:
                        t_stat, p_val = stats.ttest_ind(exp_returns, all_returns)
                    except:
                        t_stat, p_val = np.nan, np.nan
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        ((len(exp_returns) - 1) * exp_returns.std()**2 +
                         (len(all_returns) - 1) * all_returns.std()**2) /
                        (len(exp_returns) + len(all_returns) - 2)
                    )
                    cohens_d = (mean_exp - mean_all) / pooled_std if pooled_std > 0 else 0
                    
                    print(f"  {threshold_name} {minutes}min: "
                          f"{len(exp_returns)} events, "
                          f"mean={mean_exp:.6f}, "
                          f"p={p_val:.4f}, "
                          f"d={cohens_d:.3f}")
                    
                    results.append({
                        'test_period': test_period + 1,
                        'threshold': threshold_name,
                        'minutes': minutes,
                        'events': len(exp_returns),
                        'mean_expansion': mean_exp,
                        'mean_all': mean_all,
                        'effect_size': abs(cohens_d),
                        'p_value': p_val,
                        'significant': p_val < 0.05 if not np.isnan(p_val) else False,
                        'train_end_date': train_df['Date'].max(),
                        'test_start_date': test_df['Date'].min(),
                        'test_end_date': test_df['Date'].max()
                    })
        
        return pd.DataFrame(results)
    
    def walk_forward_validation(self, window_months=6, step_months=1):
        """
        Walk-forward analysis to test hypothesis stability over time
        """
        print("="*60)
        print("WALK-FORWARD VALIDATION")
        print("="*60)
        
        results = []
        
        # Convert to days for easier calculation
        window_days = window_months * 30
        step_days = step_months * 30
        
        start_date = self.df['Date'].min()
        end_date = self.df['Date'].max()
        
        current_date = start_date + timedelta(days=window_days)
        step_count = 0
        
        while current_date + timedelta(days=window_days) <= end_date:
            step_count += 1
            
            # Define train and test periods
            train_start = current_date - timedelta(days=window_days)
            train_end = current_date
            test_start = current_date
            test_end = current_date + timedelta(days=step_days)
            
            # Get data
            train_mask = (self.df['Date'] >= train_start) & (self.df['Date'] < train_end)
            test_mask = (self.df['Date'] >= test_start) & (self.df['Date'] < test_end)
            
            train_df = self.df[train_mask]
            test_df = self.df[test_mask]
            
            if len(train_df) < 1000 or len(test_df) < 100:
                current_date += timedelta(days=step_days)
                continue
            
            # Calculate thresholds on training period
            train_spread_85 = train_df['spread_close'].quantile(0.85)
            train_vol_avg = train_df['Volume'].mean()
            
            # Test expansion pattern
            expansion_mask = (
                (test_df['spread_close'] >= train_spread_85) &
                (test_df['Volume'] > train_vol_avg)
            )
            
            expansion_events = test_df[expansion_mask]
            
            if len(expansion_events) >= 5:
                # Test 60-minute returns
                exp_returns = expansion_events['return_60m'].dropna()
                all_returns = test_df['return_60m'].dropna()
                
                if len(exp_returns) >= 3 and len(all_returns) >= 10:
                    mean_exp = exp_returns.mean()
                    mean_all = all_returns.mean()
                    
                    try:
                        t_stat, p_val = stats.ttest_ind(exp_returns, all_returns)
                    except:
                        t_stat, p_val = np.nan, np.nan
                    
                    results.append({
                        'step': step_count,
                        'train_start': train_start,
                        'train_end': train_end,
                        'test_start': test_start,
                        'test_end': test_end,
                        'events': len(exp_returns),
                        'mean_expansion': mean_exp,
                        'mean_all': mean_all,
                        'p_value': p_val,
                        'significant': p_val < 0.05 if not np.isnan(p_val) else False
                    })
            
            current_date += timedelta(days=step_days)
        
        walk_df = pd.DataFrame(results)
        
        if len(walk_df) > 0:
            significant_count = walk_df['significant'].sum()
            total_count = len(walk_df)
            
            print(f"\nWalk-forward results:")
            print(f"Total steps: {total_count}")
            print(f"Significant steps: {significant_count} ({significant_count/total_count*100:.1f}%)")
            print(f"Mean effect (expansion): {walk_df['mean_expansion'].mean():.6f}")
            print(f"Mean effect (all): {walk_df['mean_all'].mean():.6f}")
        
        return walk_df
    
    def bootstrap_confidence_intervals(self, n_bootstrap=1000, out_of_sample_split=0.3):
        """
        Bootstrap analysis to get confidence intervals for effect size
        """
        print("="*60)
        print("BOOTSTRAP CONFIDENCE INTERVALS")
        print("="*60)
        
        # Split data chronologically
        split_idx = int(len(self.df) * (1 - out_of_sample_split))
        train_df = self.df.iloc[:split_idx].copy()
        test_df = self.df.iloc[split_idx:].copy()
        
        # Calculate thresholds on training data
        train_spread_85 = train_df['spread_close'].quantile(0.85)
        train_vol_avg = train_df['Volume'].mean()
        
        # Get expansion events in test data
        expansion_mask = (
            (test_df['spread_close'] >= train_spread_85) &
            (test_df['Volume'] > train_vol_avg)
        )
        
        expansion_events = test_df[expansion_mask]
        expansion_returns = expansion_events['return_60m'].dropna()
        all_returns = test_df['return_60m'].dropna()
        
        if len(expansion_returns) < 20:
            print("⚠️  Insufficient expansion events for bootstrap")
            return None
        
        print(f"Expansion events: {len(expansion_events)}")
        print(f"Available returns: {len(expansion_returns)}")
        
        # Bootstrap sampling
        bootstrap_effects = []
        
        for i in range(n_bootstrap):
            # Resample with replacement
            boot_exp = np.random.choice(expansion_returns, size=len(expansion_returns), replace=True)
            boot_all = np.random.choice(all_returns, size=len(all_returns), replace=True)
            
            effect = boot_exp.mean() - boot_all.mean()
            bootstrap_effects.append(effect)
        
        bootstrap_effects = np.array(bootstrap_effects)
        
        # Calculate confidence intervals
        ci_2_5 = np.percentile(bootstrap_effects, 2.5)
        ci_97_5 = np.percentile(bootstrap_effects, 97.5)
        mean_effect = bootstrap_effects.mean()
        
        print(f"\nBootstrap Results (n={n_bootstrap}):")
        print(f"Mean effect: {mean_effect:.6f}")
        print(f"95% CI: [{ci_2_5:.6f}, {ci_97_5:.6f}]")
        print(f"CI excludes zero: {'✅ Yes' if ci_97_5 < 0 or ci_2_5 > 0 else '❌ No'}")
        
        return {
            'mean_effect': mean_effect,
            'ci_lower': ci_2_5,
            'ci_upper': ci_97_5,
            'bootstrap_effects': bootstrap_effects
        }
    
    def comprehensive_validation(self):
        """
        Run all validation methods
        """
        print("🔬 COMPREHENSIVE VALIDATION FRAMEWORK")
        print("Testing spread expansion hypothesis with rigorous methods")
        print("="*80)
        
        results = {}
        
        # 1. Chronological validation
        chrono_results = self.chronological_validation()
        results['chronological'] = chrono_results
        
        # 2. Walk-forward validation
        walkforward_results = self.walk_forward_validation()
        results['walkforward'] = walkforward_results
        
        # 3. Bootstrap confidence intervals
        bootstrap_results = self.bootstrap_confidence_intervals()
        results['bootstrap'] = bootstrap_results
        
        # Final assessment
        print("\n" + "="*80)
        print("FINAL VALIDATION ASSESSMENT")
        print("="*80)
        
        validation_score = 0
        total_tests = 0
        
        # Assess chronological validation
        if len(chrono_results) > 0:
            significant_chrono = chrono_results['significant'].sum()
            total_chrono = len(chrono_results)
            chrono_pass_rate = significant_chrono / total_chrono
            
            print(f"✓ Chronological validation: {significant_chrono}/{total_chrono} ({chrono_pass_rate*100:.1f}%) significant")
            validation_score += chrono_pass_rate
            total_tests += 1
        
        # Assess walk-forward validation
        if len(walkforward_results) > 0:
            significant_walk = walkforward_results['significant'].sum()
            total_walk = len(walkforward_results)
            walk_pass_rate = significant_walk / total_walk if total_walk > 0 else 0
            
            print(f"✓ Walk-forward validation: {significant_walk}/{total_walk} ({walk_pass_rate*100:.1f}%) significant")
            validation_score += walk_pass_rate
            total_tests += 1
        
        # Assess bootstrap confidence
        if bootstrap_results:
            ci_excludes_zero = (bootstrap_results['ci_upper'] < 0) or (bootstrap_results['ci_lower'] > 0)
            bootstrap_pass = 1.0 if ci_excludes_zero else 0.0
            
            print(f"✓ Bootstrap confidence: CI excludes zero = {ci_excludes_zero}")
            validation_score += bootstrap_pass
            total_tests += 1
        
        # Final score
        final_score = validation_score / total_tests if total_tests > 0 else 0
        
        print(f"\n🎯 VALIDATION SCORE: {final_score:.2f} / 1.00")
        
        if final_score >= 0.7:
            print("✅ HYPOTHESIS VALIDATED - Strong evidence across multiple tests")
        elif final_score >= 0.5:
            print("⚠️  HYPOTHESIS PARTIALLY VALIDATED - Mixed evidence")
        else:
            print("❌ HYPOTHESIS NOT VALIDATED - Insufficient evidence")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if len(chrono_results) > 0:
            chrono_results.to_csv(f'validation_chronological_{timestamp}.csv', index=False)
        
        if len(walkforward_results) > 0:
            walkforward_results.to_csv(f'validation_walkforward_{timestamp}.csv', index=False)
        
        print(f"\n💾 Validation results saved with timestamp: {timestamp}")
        
        return results, final_score

def main():
    """Run comprehensive validation"""
    file_path = "storage/data/capital_com/GBPUSD/raw/1m/2024-06-01_2025-05-23.csv"
    
    validator = RigorousValidator(file_path)
    results, score = validator.comprehensive_validation()
    
    return validator, results, score

if __name__ == "__main__":
    validator, results, score = main()