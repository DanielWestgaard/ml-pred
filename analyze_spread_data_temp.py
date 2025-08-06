import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from scipy import stats

class CandlestickAnalyzer:
    def __init__(self, csv_file_path):
        """Initialize with CSV file path"""
        self.df = pd.read_csv(csv_file_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Calculate derived metrics for expansion analysis
        self._calculate_expansion_metrics()
    
    def _calculate_expansion_metrics(self):
        """Calculate metrics for spread expansion analysis"""
        # Mid price and returns
        self.df['mid_price'] = (self.df['bid_close'] + self.df['ask_close']) / 2
        
        # Forward returns for validation
        for minutes in [5, 10, 15, 30, 45, 60]:
            self.df[f'return_{minutes}m'] = (
                self.df['mid_price'].shift(-minutes) - self.df['mid_price']
            ) / self.df['mid_price']
        
        # Rolling spread percentiles (240-minute window for expansion detection)
        window = 240
        self.df['spread_90pct'] = self.df['spread_close'].rolling(window=window, min_periods=10).quantile(0.90)
        self.df['spread_85pct'] = self.df['spread_close'].rolling(window=window, min_periods=10).quantile(0.85)
        self.df['spread_80pct'] = self.df['spread_close'].rolling(window=window, min_periods=10).quantile(0.80)
        
        # Expansion flags
        self.df['expansion_90pct'] = self.df['spread_close'] >= self.df['spread_90pct']
        self.df['expansion_85pct'] = self.df['spread_close'] >= self.df['spread_85pct']
        self.df['expansion_80pct'] = self.df['spread_close'] >= self.df['spread_80pct']
        
        # Volume conditions
        self.df['volume_avg_240m'] = self.df['Volume'].rolling(window=240, min_periods=1).mean()
        self.df['volume_above_avg'] = self.df['Volume'] > self.df['volume_avg_240m']
        
        # Combined expansion signals
        self.df['expansion_signal_90'] = (
            self.df['expansion_90pct'] & 
            self.df['volume_above_avg']
        )
        self.df['expansion_signal_85'] = (
            self.df['expansion_85pct'] & 
            self.df['volume_above_avg']
        )
        self.df['expansion_signal_80'] = (
            self.df['expansion_80pct'] & 
            self.df['volume_above_avg']
        )
        
    def plot_overlay_view(self, show_volume=True):
        """Plot bid and ask candlesticks overlayed on same chart"""
        fig = make_subplots(
            rows=2 if show_volume else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Bid/Ask Prices', 'Volume') if show_volume else ('Bid/Ask Prices',),
            # row_weights=[0.7, 0.3] if show_volume else [1]
        )
        
        # Bid candlesticks
        fig.add_trace(
            go.Candlestick(
                x=self.df['Date'],
                open=self.df['bid_open'],
                high=self.df['bid_high'],
                low=self.df['bid_low'],
                close=self.df['bid_close'],
                name='Bid',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Ask candlesticks
        fig.add_trace(
            go.Candlestick(
                x=self.df['Date'],
                open=self.df['ask_open'],
                high=self.df['ask_high'],
                low=self.df['ask_low'],
                close=self.df['ask_close'],
                name='Ask',
                increasing_line_color='#42a5f5',
                decreasing_line_color='#ff7043',
                opacity=0.8
            ),
            row=1, col=1
        )
        
        if show_volume:
            # Volume bars
            colors = ['green' if close >= open else 'red' 
                     for close, open in zip(self.df['Close'], self.df['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=self.df['Date'],
                    y=self.df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Bid/Ask Candlestick Chart - Overlay View',
            xaxis_rangeslider_visible=False,
            height=600 if show_volume else 500,
            showlegend=True
        )
        
        return fig
    
    def plot_separate_view(self, show_volume=True):
        """Plot bid and ask candlesticks in separate subplots"""
        rows = 3 if show_volume else 2
        subplot_titles = ['Bid Prices', 'Ask Prices', 'Volume'] if show_volume else ['Bid Prices', 'Ask Prices']
        row_weights = [0.4, 0.4, 0.2] if show_volume else [0.5, 0.5]
        
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            # row_weights=row_weights
        )
        
        # Bid candlesticks
        fig.add_trace(
            go.Candlestick(
                x=self.df['Date'],
                open=self.df['bid_open'],
                high=self.df['bid_high'],
                low=self.df['bid_low'],
                close=self.df['bid_close'],
                name='Bid',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Ask candlesticks
        fig.add_trace(
            go.Candlestick(
                x=self.df['Date'],
                open=self.df['ask_open'],
                high=self.df['ask_high'],
                low=self.df['ask_low'],
                close=self.df['ask_close'],
                name='Ask',
                increasing_line_color='#42a5f5',
                decreasing_line_color='#ff7043'
            ),
            row=2, col=1
        )
        
        if show_volume:
            # Volume bars
            colors = ['green' if close >= open else 'red' 
                     for close, open in zip(self.df['Close'], self.df['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=self.df['Date'],
                    y=self.df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.6
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title='Bid/Ask Candlestick Chart - Separate View',
            xaxis_rangeslider_visible=False,
            height=700 if show_volume else 600,
            showlegend=True
        )
        
        return fig
    
    def plot_spread_analysis(self):
        """Plot spread analysis"""
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=['Price Comparison', 'Spread Analysis']
        )
        
        # Price comparison
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'],
                y=self.df['bid_close'],
                name='Bid Close',
                line=dict(color='#26a69a', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'],
                y=self.df['ask_close'],
                name='Ask Close',
                line=dict(color='#42a5f5', width=2)
            ),
            row=1, col=1
        )
        
        # Fill between bid and ask
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'].tolist() + self.df['Date'].tolist()[::-1],
                y=self.df['bid_close'].tolist() + self.df['ask_close'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(100, 100, 100, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Spread',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Spread analysis
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'],
                y=self.df['spread_close'],
                name='Close Spread',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'],
                y=self.df['spread_avg'],
                name='Average Spread',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Spread Analysis',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_single_price(self, price_type='bid'):
        """Plot single price type (bid or ask) candlesticks"""
        prefix = price_type.lower()
        
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{price_type.title()} Prices', 'Volume'),
            # row_weights=[0.7, 0.3]
        )
        
        fig.add_trace(
            go.Candlestick(
                x=self.df['Date'],
                open=self.df[f'{prefix}_open'],
                high=self.df[f'{prefix}_high'],
                low=self.df[f'{prefix}_low'],
                close=self.df[f'{prefix}_close'],
                name=f'{price_type.title()} Price',
                increasing_line_color='#26a69a' if prefix == 'bid' else '#42a5f5',
                decreasing_line_color='#ef5350' if prefix == 'bid' else '#ff7043'
            ),
            row=1, col=1
        )
        
        # Volume
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(self.df[f'{prefix}_close'], self.df[f'{prefix}_open'])]
        
        fig.add_trace(
            go.Bar(
                x=self.df['Date'],
                y=self.df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.6
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{price_type.title()} Candlestick Chart',
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_expansion_analysis(self, sample_days=7):
        """Plot spread expansion patterns with signals"""
        # Sample recent data for visualization
        sample_df = self.df.tail(sample_days * 1440).copy()  # Last N days of 1-minute data
        
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[
                'Price Movement with Expansion Signals',
                'Spread Analysis with Expansion Thresholds', 
                'Volume and Volume Average',
                'Forward Returns After Expansion Events'
            ],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}], 
                   [{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # 1. Price with expansion signals
        fig.add_trace(
            go.Scatter(
                x=sample_df['Date'],
                y=sample_df['mid_price'],
                name='Mid Price',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Add expansion signals as markers
        for signal, color, name in [
            ('expansion_signal_90', 'red', '90th Pct Expansion'),
            ('expansion_signal_85', 'orange', '85th Pct Expansion'),
            ('expansion_signal_80', 'yellow', '80th Pct Expansion')
        ]:
            expansion_events = sample_df[sample_df[signal]]
            if len(expansion_events) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=expansion_events['Date'],
                        y=expansion_events['mid_price'],
                        mode='markers',
                        name=name,
                        marker=dict(color=color, size=8, symbol='triangle-up')
                    ),
                    row=1, col=1
                )
        
        # 2. Spread with thresholds
        fig.add_trace(
            go.Scatter(
                x=sample_df['Date'],
                y=sample_df['spread_close'],
                name='Spread',
                line=dict(color='black', width=2)
            ),
            row=2, col=1
        )
        
        # Add percentile thresholds
        for threshold, color, name in [
            ('spread_90pct', 'red', '90th Percentile'),
            ('spread_85pct', 'orange', '85th Percentile'),
            ('spread_80pct', 'gold', '80th Percentile')
        ]:
            fig.add_trace(
                go.Scatter(
                    x=sample_df['Date'],
                    y=sample_df[threshold],
                    name=name,
                    line=dict(color=color, width=1, dash='dash'),
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # 3. Volume analysis
        fig.add_trace(
            go.Bar(
                x=sample_df['Date'],
                y=sample_df['Volume'],
                name='Volume',
                marker_color='lightblue',
                opacity=0.6
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sample_df['Date'],
                y=sample_df['volume_avg_240m'],
                name='Volume Average (240m)',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        
        # 4. Forward returns analysis
        # Show 60-minute forward returns
        fig.add_trace(
            go.Scatter(
                x=sample_df['Date'],
                y=sample_df['return_60m'] * 10000,  # Convert to basis points
                name='60min Forward Return (bp)',
                line=dict(color='purple', width=1),
                opacity=0.7
            ),
            row=4, col=1
        )
        
        # Add zero line for reference
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
        
        fig.update_layout(
            height=1000,
            title_text=f"Spread Expansion Analysis - Last {sample_days} Days",
            showlegend=True
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Spread", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        fig.update_yaxes(title_text="Return (bp)", row=4, col=1)
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
        return fig
    
    def validate_expansion_hypothesis(self, out_of_sample_split=0.3):
        """Validate expansion hypothesis with proper out-of-sample testing"""
        print("=== RIGOROUS VALIDATION: SPREAD EXPANSION HYPOTHESIS ===")
        
        # Split data chronologically to avoid lookahead bias
        split_idx = int(len(self.df) * (1 - out_of_sample_split))
        train_df = self.df.iloc[:split_idx].copy()
        test_df = self.df.iloc[split_idx:].copy()
        
        print(f"Training period: {train_df['Date'].min()} to {train_df['Date'].max()}")
        print(f"Testing period: {test_df['Date'].min()} to {test_df['Date'].max()}")
        print(f"Training samples: {len(train_df):,}")
        print(f"Testing samples: {len(test_df):,}")
        
        # Calculate percentiles ONLY on training data
        window = 240
        train_90pct = train_df['spread_close'].quantile(0.90)
        train_85pct = train_df['spread_close'].quantile(0.85)
        train_80pct = train_df['spread_close'].quantile(0.80)
        train_vol_avg = train_df['Volume'].mean()
        
        print(f"\nTraining data thresholds:")
        print(f"90th percentile spread: {train_90pct:.6f}")
        print(f"85th percentile spread: {train_85pct:.6f}")
        print(f"80th percentile spread: {train_80pct:.6f}")
        print(f"Average volume: {train_vol_avg:.1f}")
        
        # Apply thresholds to test data (NO LOOKAHEAD)
        results = []
        
        for threshold_name, threshold_value in [
            ('90th_pct', train_90pct),
            ('85th_pct', train_85pct),
            ('80th_pct', train_80pct)
        ]:
            # Define expansion condition using TRAINING thresholds
            expansion_condition = (
                (test_df['spread_close'] >= threshold_value) &
                (test_df['Volume'] > train_vol_avg)
            )
            
            expansion_events = test_df[expansion_condition]
            
            if len(expansion_events) < 10:
                continue
                
            print(f"\n{threshold_name} Expansion Events: {len(expansion_events)}")
            
            # Test forward returns
            for minutes in [45, 60]:
                return_col = f'return_{minutes}m'
                
                if return_col not in expansion_events.columns:
                    continue
                
                expansion_returns = expansion_events[return_col].dropna()
                all_returns = test_df[return_col].dropna()
                
                if len(expansion_returns) < 5:
                    continue
                
                # Statistical tests
                mean_expansion = expansion_returns.mean()
                mean_all = all_returns.mean()
                std_expansion = expansion_returns.std()
                
                # T-test
                try:
                    t_stat, p_value = stats.ttest_ind(expansion_returns, all_returns)
                except:
                    t_stat, p_value = np.nan, np.nan
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(expansion_returns) - 1) * std_expansion**2 + 
                                    (len(all_returns) - 1) * all_returns.std()**2) / 
                                   (len(expansion_returns) + len(all_returns) - 2))
                cohens_d = (mean_expansion - mean_all) / pooled_std if pooled_std > 0 else 0
                
                # Direction bias
                positive_pct = (expansion_returns > 0).mean() * 100
                
                print(f"  {minutes}min forward:")
                print(f"    Mean return: {mean_expansion:.6f} vs {mean_all:.6f} (all)")
                print(f"    Effect size (Cohen's d): {cohens_d:.3f}")
                print(f"    P-value: {p_value:.4f}")
                print(f"    Direction: {positive_pct:.1f}% positive")
                
                results.append({
                    'threshold': threshold_name,
                    'minutes': minutes,
                    'events': len(expansion_events),
                    'mean_expansion': mean_expansion,
                    'mean_all': mean_all,
                    'effect_size': abs(cohens_d),
                    'p_value': p_value,
                    'positive_pct': positive_pct,
                    'significant': p_value < 0.05 if not np.isnan(p_value) else False
                })
        
        validation_df = pd.DataFrame(results)
        
        # Summary
        if len(validation_df) > 0:
            significant_results = validation_df[validation_df['significant']]
            print(f"\n=== OUT-OF-SAMPLE VALIDATION RESULTS ===")
            print(f"Total patterns tested: {len(validation_df)}")
            print(f"Statistically significant: {len(significant_results)}")
            
            if len(significant_results) > 0:
                print("✅ HYPOTHESIS VALIDATED - Significant patterns found in out-of-sample data:")
                for _, row in significant_results.iterrows():
                    print(f"  • {row['threshold']} {row['minutes']}min: "
                          f"Effect size {row['effect_size']:.3f}, P={row['p_value']:.4f}")
            else:
                print("❌ HYPOTHESIS NOT VALIDATED - No significant patterns in out-of-sample data")
        
        return validation_df
    
    def print_summary(self):
        """Print data summary"""
        print("=== Dataset Summary ===")
        print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        print(f"Total records: {len(self.df)}")
        print(f"Columns: {list(self.df.columns)}")
        print("\n=== Price Summary ===")
        print(f"Bid Close - Min: {self.df['bid_close'].min():.6f}, Max: {self.df['bid_close'].max():.6f}")
        print(f"Ask Close - Min: {self.df['ask_close'].min():.6f}, Max: {self.df['ask_close'].max():.6f}")
        print(f"Average Spread: {self.df['spread_avg'].mean():.6f}")
        print(f"Volume - Min: {self.df['Volume'].min():,}, Max: {self.df['Volume'].max():,}")

# Usage Example
def main():
    # Choose dataset
    large_file = 'storage/data/capital_com/GBPUSD/raw/1m/2024-06-01_2025-05-23.csv'
    small_file = 'storage/data/capital_com/GBPUSD/raw/1m/2025-07-02_2025-07-31.csv'
    
    print("Choose dataset for analysis:")
    print("1. Large dataset (2024-2025) - Recommended for validation")
    print("2. Small dataset (July 2025) - Quick visualization")
    
    choice = input("Enter choice (1 or 2, or press Enter for large): ").strip()
    file_path = large_file if choice != "2" else small_file
    
    # Initialize analyzer
    analyzer = CandlestickAnalyzer(file_path)
    
    # Print summary
    analyzer.print_summary()
    
    print("\nChoose analysis type:")
    print("1. Original spread analysis (compression focus)")
    print("2. New expansion analysis with validation")
    print("3. Both analyses")
    
    analysis_choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if analysis_choice in ["1", "3"]:
        print("\n" + "="*60)
        print("ORIGINAL SPREAD ANALYSIS")
        print("="*60)
        
        # Original analysis views
        fig1 = analyzer.plot_overlay_view(show_volume=True)
        fig1.show()
        
        fig2 = analyzer.plot_spread_analysis()
        fig2.show()
    
    if analysis_choice in ["2", "3"]:
        print("\n" + "="*60)
        print("SPREAD EXPANSION ANALYSIS & VALIDATION")
        print("="*60)
        
        # New expansion analysis
        fig3 = analyzer.plot_expansion_analysis(sample_days=14)  # Last 14 days
        fig3.show()
        
        # Rigorous out-of-sample validation
        validation_results = analyzer.validate_expansion_hypothesis(out_of_sample_split=0.3)
        
        if len(validation_results) > 0:
            print(f"\n💾 Validation results summary:")
            print(validation_results.to_string(index=False))
            
            # Save validation results
            validation_results.to_csv('expansion_validation_results.csv', index=False)
            print(f"💾 Saved validation results to: expansion_validation_results.csv")

if __name__ == "__main__":
    main()

# Alternative: Quick plotting functions for specific needs
def quick_overlay_plot(csv_file):
    """Quick function to create overlay plot"""
    analyzer = CandlestickAnalyzer(csv_file)
    fig = analyzer.plot_overlay_view()
    fig.show()
    return analyzer

def quick_separate_plot(csv_file):
    """Quick function to create separate plots"""
    analyzer = CandlestickAnalyzer(csv_file)
    fig = analyzer.plot_separate_view()
    fig.show()
    return analyzer