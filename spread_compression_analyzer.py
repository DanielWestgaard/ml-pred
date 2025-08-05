import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SpreadCompressionAnalyzer:
    """
    Comprehensive analyzer for spread compression patterns and their predictive power.
    Tests the hypothesis: "When spread compresses to low percentiles with above-average volume,
    does it predict directional price movements in the next 5-15 minutes?"
    """
    
    def __init__(self, csv_file_path):
        """Initialize with CSV file path"""
        print(f"Loading data from: {csv_file_path}")
        self.df = pd.read_csv(csv_file_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Calculate additional metrics
        self._calculate_derived_metrics()
        self._identify_market_hours()
        
        print(f"Loaded {len(self.df)} records from {self.df['Date'].min()} to {self.df['Date'].max()}")
    
    def _calculate_derived_metrics(self):
        """Calculate additional spread and price metrics"""
        # Mid price for directional analysis
        self.df['mid_price'] = (self.df['bid_close'] + self.df['ask_close']) / 2
        
        # Price changes for forward returns
        for minutes in [5, 10, 15]:
            self.df[f'price_change_{minutes}m'] = self.df['mid_price'].shift(-minutes) - self.df['mid_price']
            self.df[f'return_{minutes}m'] = self.df[f'price_change_{minutes}m'] / self.df['mid_price']
            
        # Rolling volume averages
        for window in [60, 240, 1440]:  # 1h, 4h, 24h
            self.df[f'volume_avg_{window}m'] = self.df['Volume'].rolling(window=window, min_periods=1).mean()
            self.df[f'volume_above_avg_{window}m'] = self.df['Volume'] > self.df[f'volume_avg_{window}m']
    
    def _identify_market_hours(self):
        """Identify market sessions and trading hours"""
        self.df['hour'] = self.df['Date'].dt.hour
        self.df['day_of_week'] = self.df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])  # Saturday, Sunday
        
        # London session: 8-17 UTC, NY session: 13-22 UTC, Overlap: 13-17 UTC
        self.df['is_london_session'] = (self.df['hour'] >= 8) & (self.df['hour'] < 17)
        self.df['is_ny_session'] = (self.df['hour'] >= 13) & (self.df['hour'] < 22)
        self.df['is_overlap_session'] = (self.df['hour'] >= 13) & (self.df['hour'] < 17)
        self.df['is_asian_session'] = (self.df['hour'] >= 22) | (self.df['hour'] < 8)
    
    def calculate_spread_percentiles(self, windows=[60, 240, 1440], percentiles=[5, 10, 15, 20, 25]):
        """Calculate rolling spread percentiles across different timeframes"""
        print("Calculating spread percentiles...")
        
        results = {}
        for window in windows:
            window_results = {}
            
            # Calculate rolling percentiles for different spread metrics
            for spread_col in ['spread_close', 'spread_avg']:
                for p in percentiles:
                    col_name = f'{spread_col}_pct{p}_{window}m'
                    self.df[col_name] = self.df[spread_col].rolling(window=window, min_periods=10).quantile(p/100)
                    
                    # Flag when current spread is below this percentile
                    flag_col = f'{spread_col}_below_pct{p}_{window}m'
                    self.df[flag_col] = self.df[spread_col] <= self.df[col_name]
                    
                    window_results[p] = {
                        'percentile_col': col_name,
                        'flag_col': flag_col,
                        'compression_count': self.df[flag_col].sum()
                    }
            
            results[f'{window}m'] = window_results
        
        return results
    
    def analyze_compression_frequency(self, results):
        """Analyze frequency of compression events across different thresholds"""
        print("\n=== SPREAD COMPRESSION FREQUENCY ANALYSIS ===")
        
        frequency_data = []
        for window, window_data in results.items():
            print(f"\n{window} Rolling Window:")
            for percentile, data in window_data.items():
                compression_count = data['compression_count']
                total_records = len(self.df)
                frequency_pct = (compression_count / total_records) * 100
                
                print(f"  {percentile}th percentile: {compression_count:,} events ({frequency_pct:.2f}%)")
                
                frequency_data.append({
                    'window': window,
                    'percentile': percentile,
                    'count': compression_count,
                    'frequency_pct': frequency_pct
                })
        
        return pd.DataFrame(frequency_data)
    
    def test_predictive_power(self, percentile_thresholds=[5, 10, 15, 20], 
                            windows=[240], volume_windows=[240]):
        """Test if spread compression predicts future price movements"""
        print("\n=== PREDICTIVE POWER ANALYSIS ===")
        
        results = []
        
        for window in windows:
            for vol_window in volume_windows:
                for threshold in percentile_thresholds:
                    # Define compression condition
                    compression_col = f'spread_close_below_pct{threshold}_{window}m'
                    volume_col = f'volume_above_avg_{vol_window}m'
                    
                    if compression_col not in self.df.columns:
                        continue
                    
                    # Combined condition: spread compression + above average volume
                    condition = self.df[compression_col] & self.df[volume_col] & (~self.df['is_weekend'])
                    compression_events = self.df[condition]
                    
                    if len(compression_events) < 10:  # Need minimum events for analysis
                        continue
                    
                    print(f"\nTesting {threshold}th percentile, {window}m window, {vol_window}m volume:")
                    print(f"  Found {len(compression_events)} compression events")
                    
                    # Analyze forward returns
                    for minutes in [5, 10, 15]:
                        return_col = f'return_{minutes}m'
                        
                        if return_col not in compression_events.columns:
                            continue
                        
                        # Get returns for compression events vs all periods
                        compression_returns = compression_events[return_col].dropna()
                        all_returns = self.df[return_col].dropna()
                        
                        if len(compression_returns) < 5:
                            continue
                        
                        # Statistical tests
                        mean_compression = compression_returns.mean()
                        mean_all = all_returns.mean()
                        std_compression = compression_returns.std()
                        
                        # T-test for difference in means
                        try:
                            t_stat, p_value = stats.ttest_ind(compression_returns, all_returns)
                        except:
                            t_stat, p_value = np.nan, np.nan
                        
                        # Direction bias
                        positive_pct = (compression_returns > 0).mean() * 100
                        negative_pct = (compression_returns < 0).mean() * 100
                        
                        print(f"    {minutes}min forward returns:")
                        print(f"      Mean: {mean_compression:.6f} vs {mean_all:.6f} (all periods)")
                        print(f"      Std:  {std_compression:.6f}")
                        print(f"      Direction: {positive_pct:.1f}% positive, {negative_pct:.1f}% negative")
                        print(f"      T-test p-value: {p_value:.4f}")
                        
                        results.append({
                            'percentile_threshold': threshold,
                            'window_minutes': window,
                            'volume_window': vol_window,
                            'forward_minutes': minutes,
                            'compression_events': len(compression_events),
                            'mean_return_compression': mean_compression,
                            'mean_return_all': mean_all,
                            'std_return_compression': std_compression,
                            'positive_direction_pct': positive_pct,
                            'negative_direction_pct': negative_pct,
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'statistically_significant': p_value < 0.05 if not np.isnan(p_value) else False
                        })
        
        return pd.DataFrame(results)
    
    def analyze_market_regime_impact(self):
        """Analyze how spread compression varies across market regimes"""
        print("\n=== MARKET REGIME ANALYSIS ===")
        
        regimes = {
            'London Session': 'is_london_session',
            'NY Session': 'is_ny_session', 
            'Overlap Session': 'is_overlap_session',
            'Asian Session': 'is_asian_session',
            'Weekend': 'is_weekend'
        }
        
        regime_stats = []
        
        for regime_name, regime_col in regimes.items():
            regime_data = self.df[self.df[regime_col]]
            
            if len(regime_data) == 0:
                continue
            
            stats_dict = {
                'regime': regime_name,
                'total_periods': len(regime_data),
                'avg_spread': regime_data['spread_close'].mean(),
                'std_spread': regime_data['spread_close'].std(),
                'avg_volume': regime_data['Volume'].mean(),
                'std_volume': regime_data['Volume'].std()
            }
            
            # Calculate compression frequencies for this regime
            if 'spread_close_below_pct10_240m' in regime_data.columns:
                stats_dict['compression_10pct_count'] = regime_data['spread_close_below_pct10_240m'].sum()
                stats_dict['compression_10pct_freq'] = (stats_dict['compression_10pct_count'] / len(regime_data)) * 100
            
            regime_stats.append(stats_dict)
            
            print(f"\n{regime_name}:")
            print(f"  Periods: {stats_dict['total_periods']:,}")
            print(f"  Avg Spread: {stats_dict['avg_spread']:.6f} ± {stats_dict['std_spread']:.6f}")
            print(f"  Avg Volume: {stats_dict['avg_volume']:,.0f} ± {stats_dict['std_volume']:,.0f}")
            if 'compression_10pct_freq' in stats_dict:
                print(f"  10th pct compression: {stats_dict['compression_10pct_count']} ({stats_dict['compression_10pct_freq']:.2f}%)")
        
        return pd.DataFrame(regime_stats)
    
    def plot_comprehensive_analysis(self, percentile_results):
        """Create comprehensive visualization of spread compression analysis"""
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Spread Distribution by Market Regime', 'Compression Event Frequency',
                'Forward Returns by Compression Threshold', 'Volume vs Spread Relationship',
                'Time Series: Spread with Compression Events', 'Statistical Significance Heatmap',
                'Directional Bias Analysis', 'Risk-Adjusted Returns'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.08
        )
        
        # 1. Spread distribution by market regime
        regimes = ['is_london_session', 'is_ny_session', 'is_asian_session', 'is_weekend']
        regime_names = ['London', 'NY', 'Asian', 'Weekend']
        
        for i, (regime_col, regime_name) in enumerate(zip(regimes, regime_names)):
            if regime_col in self.df.columns:
                regime_data = self.df[self.df[regime_col]]['spread_close']
                fig.add_trace(
                    go.Histogram(x=regime_data, name=regime_name, opacity=0.7, nbinsx=50),
                    row=1, col=1
                )
        
        # 2. Compression event frequency
        freq_data = []
        for window in ['60m', '240m', '1440m']:
            if window in percentile_results:
                for pct, data in percentile_results[window].items():
                    freq_data.append({
                        'window': window,
                        'percentile': pct,
                        'frequency': (data['compression_count'] / len(self.df)) * 100
                    })
        
        if freq_data:
            freq_df = pd.DataFrame(freq_data)
            for window in freq_df['window'].unique():
                window_data = freq_df[freq_df['window'] == window]
                fig.add_trace(
                    go.Scatter(x=window_data['percentile'], y=window_data['frequency'], 
                             name=f'Frequency {window}', mode='lines+markers'),
                    row=1, col=2
                )
        
        # 3. Time series with compression events
        # Sample last 1000 points for readability
        sample_df = self.df.tail(1000).copy()
        
        fig.add_trace(
            go.Scatter(x=sample_df['Date'], y=sample_df['spread_close'],
                      name='Spread', line=dict(color='blue', width=1)),
            row=3, col=1
        )
        
        # Highlight compression events if available
        if 'spread_close_below_pct10_240m' in sample_df.columns:
            compression_events = sample_df[sample_df['spread_close_below_pct10_240m']]
            if len(compression_events) > 0:
                fig.add_trace(
                    go.Scatter(x=compression_events['Date'], y=compression_events['spread_close'],
                              mode='markers', name='10th Pct Compression',
                              marker=dict(color='red', size=8, symbol='circle')),
                    row=3, col=1
                )
        
        # 4. Volume vs Spread scatter
        sample_size = min(5000, len(self.df))
        sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
        sample_data = self.df.iloc[sample_indices]
        
        fig.add_trace(
            go.Scatter(x=sample_data['Volume'], y=sample_data['spread_close'],
                      mode='markers', name='Volume vs Spread',
                      marker=dict(size=3, opacity=0.6),
                      showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(
            height=1200,
            title_text="Comprehensive Spread Compression Analysis",
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Spread", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Percentile Threshold", row=1, col=2)
        fig.update_yaxes(title_text="Compression Frequency (%)", row=1, col=2)
        fig.update_xaxes(title_text="Volume", row=2, col=2)
        fig.update_yaxes(title_text="Spread", row=2, col=2)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Spread", row=3, col=1)
        
        return fig
    
    def run_complete_analysis(self):
        """Run the complete spread compression analysis pipeline"""
        print("=== STARTING COMPREHENSIVE SPREAD COMPRESSION ANALYSIS ===")
        print(f"Dataset: {len(self.df)} records from {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        # 1. Calculate spread percentiles
        percentile_results = self.calculate_spread_percentiles()
        
        # 2. Analyze compression frequency
        frequency_df = self.analyze_compression_frequency(percentile_results)
        
        # 3. Test predictive power
        predictive_results = self.test_predictive_power()
        
        # 4. Market regime analysis
        regime_stats = self.analyze_market_regime_impact()
        
        # 5. Create comprehensive visualization
        if len(percentile_results) > 0:
            fig = self.plot_comprehensive_analysis(percentile_results)
        else:
            fig = None
        
        # 6. Summary and recommendations
        self._print_summary_recommendations(frequency_df, predictive_results, regime_stats)
        
        return {
            'percentile_results': percentile_results,
            'frequency_analysis': frequency_df,
            'predictive_results': predictive_results,
            'regime_analysis': regime_stats,
            'visualization': fig
        }
    
    def _print_summary_recommendations(self, frequency_df, predictive_results, regime_stats):
        """Print summary and recommendations"""
        print("\n" + "="*60)
        print("SUMMARY AND RECOMMENDATIONS")
        print("="*60)
        
        if len(predictive_results) > 0:
            # Find best performing configurations
            significant_results = predictive_results[predictive_results['statistically_significant'] == True]
            
            if len(significant_results) > 0:
                print(f"\n✅ FOUND {len(significant_results)} STATISTICALLY SIGNIFICANT PATTERNS:")
                for _, row in significant_results.iterrows():
                    print(f"  • {row['percentile_threshold']}th percentile, {row['forward_minutes']}min forward:")
                    print(f"    Mean return: {row['mean_return_compression']:.6f} ({row['compression_events']} events)")
                    print(f"    Direction bias: {row['positive_direction_pct']:.1f}% positive")
                    print(f"    P-value: {row['p_value']:.4f}")
            else:
                print("\n❌ NO STATISTICALLY SIGNIFICANT PATTERNS FOUND")
                print("   Consider:")
                print("   - Using different percentile thresholds")
                print("   - Longer observation periods")
                print("   - Different volume conditions")
                print("   - Alternative spread compression definitions")
        
        # Dataset recommendations
        total_events_10pct = frequency_df[frequency_df['percentile'] == 10]['count'].sum()
        print(f"\n📊 DATASET INSIGHTS:")
        print(f"   • 10th percentile compressions: {total_events_10pct} total events")
        print(f"   • Recommendation: {'✅ Sufficient data' if total_events_10pct > 100 else '⚠️  Consider larger dataset'}")
        
        if len(regime_stats) > 0:
            weekend_spread = regime_stats[regime_stats['regime'] == 'Weekend']['avg_spread'].iloc[0] if 'Weekend' in regime_stats['regime'].values else 0
            london_spread = regime_stats[regime_stats['regime'] == 'London Session']['avg_spread'].iloc[0] if 'London Session' in regime_stats['regime'].values else 0
            
            if weekend_spread > london_spread * 2:
                print(f"   • Weekend spreads are {weekend_spread/london_spread:.1f}x higher - consider filtering")

def main():
    """Main analysis function"""
    # Test on larger dataset first
    large_file = "storage/data/capital_com/GBPUSD/raw/1m/2024-06-01_2025-05-23.csv"
    small_file = "storage/data/capital_com/GBPUSD/raw/1m/2025-07-02_2025-07-31.csv"
    
    # Use large dataset by default for comprehensive analysis
    file_path = large_file
    print(f"Using large dataset: {file_path}")
    
    try:
        # Initialize analyzer
        analyzer = SpreadCompressionAnalyzer(file_path)
        
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        # Show visualization if created
        if results['visualization']:
            results['visualization'].show()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if len(results['predictive_results']) > 0:
            results['predictive_results'].to_csv(f'spread_analysis_predictive_{timestamp}.csv', index=False)
            print(f"\n💾 Saved predictive results to: spread_analysis_predictive_{timestamp}.csv")
        
        if len(results['frequency_analysis']) > 0:
            results['frequency_analysis'].to_csv(f'spread_analysis_frequency_{timestamp}.csv', index=False)
            print(f"💾 Saved frequency analysis to: spread_analysis_frequency_{timestamp}.csv")
        
        return analyzer, results
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Please check the file path and try again.")
        return None, None
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return None, None

if __name__ == "__main__":
    analyzer, results = main()