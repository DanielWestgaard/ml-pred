import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class CandlestickAnalyzer:
    def __init__(self, csv_file_path):
        """Initialize with CSV file path"""
        self.df = pd.read_csv(csv_file_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date')
        
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
    # Initialize analyzer with your CSV file
    analyzer = CandlestickAnalyzer('storage/data/capital_com/GBPUSD/raw/1m/2025-07-02_2025-07-31.csv')  # Replace with your file path
    
    # Print summary
    analyzer.print_summary()
    
    # Create different views
    
    # 1. Overlay view (bid and ask on same chart)
    fig1 = analyzer.plot_overlay_view(show_volume=True)
    fig1.show()
    
    # 2. Separate view (bid and ask in separate subplots)
    fig2 = analyzer.plot_separate_view(show_volume=True)
    fig2.show()
    
    # 3. Spread analysis
    fig3 = analyzer.plot_spread_analysis()
    fig3.show()
    
    # 4. Individual price views
    fig4 = analyzer.plot_single_price('bid')
    fig4.show()
    
    fig5 = analyzer.plot_single_price('ask')
    fig5.show()

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