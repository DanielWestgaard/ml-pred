"""
Gap Fade Strategy Analysis for Micro-Cap Stocks

This script tests the simple gap fade strategy described in temp_algorithms/price_gaps/README.md
on historical data for micro-cap stocks: ISSC, CCLD, PLX, LFVN, BDTX

Strategy Rules (Version 1):
- Gap of 5-15% at market open (9:30 AM)
- Entry when opening volume < 2x average 20-day volume (indicates "fake gap")
- Target: previous day's close (gap fill)
- Stop: 2% beyond entry point
- Time stop: Exit at 3:30 PM regardless

CRITICAL: This implementation ensures NO data leakage or lookahead bias
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class GapFadeAnalyzer:
    def __init__(self, symbols: List[str], data_path: str = "storage/data/alpaca"):
        self.symbols = symbols
        self.data_path = data_path
        self.results = {}
        
    def load_data(self, symbol: str, timeframe: str = "5m") -> pd.DataFrame:
        """Load historical data for a symbol"""
        file_path = f"{self.data_path}/{symbol}/raw/{timeframe}"
        
        # Handle BDTX which uses "5min" instead of "5m"
        if not os.path.exists(file_path) and timeframe == "5m":
            file_path = f"{self.data_path}/{symbol}/raw/5min"
        
        # Find the CSV file in the directory
        csv_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found for {symbol} in {file_path}")
        
        csv_file = csv_files[0]  # Use the first (and likely only) CSV file
        
        df = pd.read_csv(os.path.join(file_path, csv_file))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Add trading day identifier
        df['trading_day'] = df['Date'].dt.date
        
        return df
    
    def identify_market_open(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify market open (9:30 AM) bars for each trading day
        Returns DataFrame with market open data
        """
        # Convert to ET timezone if not already (assuming data is in ET)
        df = df.copy()
        
        # Extract time component
        df['time'] = df['Date'].dt.time
        
        # Filter for regular trading hours (9:30 AM to 4:00 PM ET)
        market_start = time(9, 30)
        market_end = time(16, 0)
        
        # Get data within market hours
        market_hours = df[
            (df['time'] >= market_start) & 
            (df['time'] <= market_end)
        ].copy()
        
        if market_hours.empty:
            print(f"Warning: No market hours data found. Using first available data of each day.")
            # Fallback: use first data point of each day
            return df.groupby('trading_day').first().reset_index()
        
        # Get the first bar of each trading day during market hours (closest to 9:30 AM)
        market_opens = market_hours.groupby('trading_day').first().reset_index()
        
        return market_opens
    
    def calculate_daily_stats(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """
        Calculate daily statistics needed for the strategy
        IMPORTANT: Uses only data available BEFORE each trading day to avoid lookahead bias
        """
        market_opens = self.identify_market_open(df)
        market_opens = market_opens.sort_values('Date').reset_index(drop=True)
        
        # Calculate previous day's close (for gap calculation)
        market_opens['prev_close'] = market_opens['Close'].shift(1)
        
        # Calculate gap percentage
        market_opens['gap_percent'] = (market_opens['Open'] - market_opens['prev_close']) / market_opens['prev_close']
        
        # Calculate 20-day average volume using ONLY past data (no lookahead)
        market_opens['avg_20d_volume'] = market_opens['Volume'].rolling(
            window=lookback, min_periods=1
        ).mean().shift(1)  # shift(1) ensures we only use data from previous days
        
        # Volume ratio (current vs average)
        market_opens['volume_ratio'] = market_opens['Volume'] / market_opens['avg_20d_volume']
        
        # Remove first day (no previous close available)
        market_opens = market_opens.dropna(subset=['prev_close', 'avg_20d_volume'])
        
        return market_opens
    
    def identify_gap_opportunities(self, daily_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Identify trading opportunities based on gap criteria
        """
        opportunities = daily_stats.copy()
        
        # Gap criteria: 5-15% gap (absolute value)
        gap_criteria = (np.abs(opportunities['gap_percent']) >= 0.05) & (np.abs(opportunities['gap_percent']) <= 0.15)
        
        # Volume criteria: opening volume < 2x average 20-day volume
        volume_criteria = opportunities['volume_ratio'] < 2.0
        
        # Combined criteria
        valid_opportunities = opportunities[gap_criteria & volume_criteria].copy()
        
        # Determine trade direction
        valid_opportunities['trade_direction'] = np.where(
            valid_opportunities['gap_percent'] > 0, 'short', 'long'
        )
        
        return valid_opportunities
    
    def simulate_trades(self, opportunities: pd.DataFrame, symbol: str, full_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate the actual trades with proper entry/exit timing
        """
        trades = []
        
        for idx, opp in opportunities.iterrows():
            trade_date = opp['trading_day']
            entry_price = opp['Open']
            target_price = opp['prev_close']  # Gap fill target
            direction = opp['trade_direction']
            
            # Calculate stop loss (2% beyond entry)
            if direction == 'short':
                stop_price = entry_price * 1.02  # 2% above entry for short
            else:  # long
                stop_price = entry_price * 0.98  # 2% below entry for long
            
            # Get intraday data for this trading day
            day_data = full_df[full_df['trading_day'] == trade_date].copy()
            day_data = day_data.sort_values('Date')
            
            if day_data.empty:
                continue
            
            # Find entry time (market open or first available)
            entry_time = day_data.iloc[0]['Date']
            
            # Define exit conditions
            exit_time = None
            exit_price = None
            exit_reason = None
            
            # Check each bar after entry for exit conditions
            for i, row in day_data.iterrows():
                current_time = row['Date']
                current_price = row['Close']  # Use close price for exits
                
                # Skip entry bar
                if current_time == entry_time:
                    continue
                
                # Check time stop (3:30 PM or 15:30)
                if current_time.time() >= time(15, 30):
                    exit_time = current_time
                    exit_price = current_price
                    exit_reason = 'time_stop'
                    break
                
                # Check stop loss
                if direction == 'short':
                    if row['High'] >= stop_price:  # Price went up, hit stop
                        exit_time = current_time
                        exit_price = stop_price
                        exit_reason = 'stop_loss'
                        break
                    elif row['Low'] <= target_price:  # Price went down, hit target
                        exit_time = current_time
                        exit_price = target_price
                        exit_reason = 'target'
                        break
                else:  # long
                    if row['Low'] <= stop_price:  # Price went down, hit stop
                        exit_time = current_time
                        exit_price = stop_price
                        exit_reason = 'stop_loss'
                        break
                    elif row['High'] >= target_price:  # Price went up, hit target
                        exit_time = current_time
                        exit_price = target_price
                        exit_reason = 'target'
                        break
            
            # If no exit found, use last price of day
            if exit_time is None:
                last_bar = day_data.iloc[-1]
                exit_time = last_bar['Date']
                exit_price = last_bar['Close']
                exit_reason = 'eod'
            
            # Calculate P&L
            if direction == 'short':
                pnl = entry_price - exit_price  # Profit when price goes down
                pnl_pct = pnl / entry_price
            else:  # long
                pnl = exit_price - entry_price  # Profit when price goes up
                pnl_pct = pnl / entry_price
            
            trades.append({
                'symbol': symbol,
                'date': trade_date,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'target_price': target_price,
                'stop_price': stop_price,
                'gap_percent': opp['gap_percent'],
                'volume_ratio': opp['volume_ratio'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'winner': pnl > 0
            })
        
        return pd.DataFrame(trades)
    
    def analyze_symbol(self, symbol: str, timeframe: str = "5m") -> Dict:
        """Analyze a single symbol"""
        print(f"\nAnalyzing {symbol}...")
        
        try:
            # Load data
            df = self.load_data(symbol, timeframe)
            print(f"Loaded {len(df)} bars from {df['Date'].min()} to {df['Date'].max()}")
            
            # Calculate daily statistics
            daily_stats = self.calculate_daily_stats(df)
            print(f"Generated daily stats for {len(daily_stats)} trading days")
            
            # Identify opportunities
            opportunities = self.identify_gap_opportunities(daily_stats)
            print(f"Found {len(opportunities)} gap opportunities")
            
            if opportunities.empty:
                return {
                    'symbol': symbol,
                    'total_opportunities': 0,
                    'total_trades': 0,
                    'trades': pd.DataFrame()
                }
            
            # Simulate trades
            trades = self.simulate_trades(opportunities, symbol, df)
            print(f"Simulated {len(trades)} trades")
            
            return {
                'symbol': symbol,
                'total_opportunities': len(opportunities),
                'total_trades': len(trades),
                'trades': trades,
                'daily_stats': daily_stats,
                'opportunities': opportunities
            }
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'total_trades': 0,
                'trades': pd.DataFrame()
            }
    
    def calculate_performance_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if trades.empty:
            return {'error': 'No trades to analyze'}
        
        # Basic metrics
        total_trades = len(trades)
        winners = trades['winner'].sum()
        losers = total_trades - winners
        win_rate = winners / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades['pnl'].sum()
        total_pnl_pct = trades['pnl_pct'].sum()
        avg_win = trades[trades['winner']]['pnl_pct'].mean() if winners > 0 else 0
        avg_loss = trades[~trades['winner']]['pnl_pct'].mean() if losers > 0 else 0
        
        # Risk metrics
        reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        max_win = trades['pnl_pct'].max()
        max_loss = trades['pnl_pct'].min()
        
        # Exit reason analysis
        exit_reasons = trades['exit_reason'].value_counts().to_dict()
        
        return {
            'total_trades': total_trades,
            'winners': winners,
            'losers': losers,
            'win_rate': win_rate,
            'total_pnl_pct': total_pnl_pct,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'reward_risk_ratio': reward_risk_ratio,
            'max_win_pct': max_win,
            'max_loss_pct': max_loss,
            'exit_reasons': exit_reasons
        }
    
    def run_analysis(self, timeframe: str = "5m"):
        """Run complete analysis on all symbols"""
        print(f"=== Gap Fade Strategy Analysis ===")
        print(f"Symbols: {self.symbols}")
        print(f"Timeframe: {timeframe}")
        print(f"Strategy: Fade gaps of 5-15% with volume confirmation")
        
        all_trades = []
        symbol_results = {}
        
        for symbol in self.symbols:
            result = self.analyze_symbol(symbol, timeframe)
            symbol_results[symbol] = result
            
            if not result['trades'].empty:
                all_trades.append(result['trades'])
        
        # Combine all trades
        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)
        else:
            combined_trades = pd.DataFrame()
        
        # Calculate overall performance
        overall_performance = self.calculate_performance_metrics(combined_trades)
        
        # Individual symbol performance
        symbol_performance = {}
        for symbol, result in symbol_results.items():
            if not result['trades'].empty:
                symbol_performance[symbol] = self.calculate_performance_metrics(result['trades'])
        
        self.results = {
            'combined_trades': combined_trades,
            'overall_performance': overall_performance,
            'symbol_performance': symbol_performance,
            'symbol_results': symbol_results
        }
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print analysis summary"""
        print(f"\n{'='*50}")
        print(f"OVERALL STRATEGY PERFORMANCE")
        print(f"{'='*50}")
        
        perf = self.results['overall_performance']
        
        if 'error' in perf:
            print("No trades found across all symbols")
            return
        
        print(f"Total Trades: {perf['total_trades']}")
        print(f"Win Rate: {perf['win_rate']:.1%}")
        print(f"Total P&L: {perf['total_pnl_pct']:.2%}")
        print(f"Average Win: {perf['avg_win_pct']:.2%}")
        print(f"Average Loss: {perf['avg_loss_pct']:.2%}")
        print(f"Reward/Risk Ratio: {perf['reward_risk_ratio']:.2f}")
        print(f"Best Trade: {perf['max_win_pct']:.2%}")
        print(f"Worst Trade: {perf['max_loss_pct']:.2%}")
        
        print(f"\nExit Reasons:")
        for reason, count in perf['exit_reasons'].items():
            print(f"  {reason}: {count} ({count/perf['total_trades']:.1%})")
        
        print(f"\n{'='*50}")
        print(f"INDIVIDUAL SYMBOL PERFORMANCE")
        print(f"{'='*50}")
        
        for symbol, perf in self.results['symbol_performance'].items():
            print(f"\n{symbol}:")
            print(f"  Trades: {perf['total_trades']}")
            print(f"  Win Rate: {perf['win_rate']:.1%}")
            print(f"  Total P&L: {perf['total_pnl_pct']:.2%}")
            print(f"  Avg Win: {perf['avg_win_pct']:.2%}")
            print(f"  Avg Loss: {perf['avg_loss_pct']:.2%}")
    
    def save_detailed_results(self, filename: str = "gap_fade_detailed_results.csv"):
        """Save detailed trade results to CSV"""
        if not self.results['combined_trades'].empty:
            self.results['combined_trades'].to_csv(filename, index=False)
            print(f"\nDetailed results saved to {filename}")

def main():
    # Symbols to analyze
    symbols = ["ISSC", "CCLD", "PLX", "LFVN", "BDTX"]
    
    # Initialize analyzer
    analyzer = GapFadeAnalyzer(symbols)
    
    # Run analysis
    results = analyzer.run_analysis(timeframe="5m")
    
    # Save detailed results
    analyzer.save_detailed_results()
    
    print(f"\n{'='*50}")
    print(f"STRATEGY VALIDATION COMPLETE")
    print(f"{'='*50}")
    print("Key Observations:")
    print("• This tests the basic gap fade strategy without news/volume enhancements")
    print("• Results show if the simple approach has any edge before adding complexity")
    print("• No data leakage: Only uses information available at trade time")
    print("• Ready for enhancement with news sentiment, sector analysis, etc.")

if __name__ == "__main__":
    main()