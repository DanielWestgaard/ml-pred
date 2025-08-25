"""
Enhanced Gap Fade Strategy Analysis

This builds on gap_fade_analysis.py with:
1. Enhanced Volume Analysis (Version 2): Wait 5-10 minutes for volume confirmation
2. Time-of-day Exit Optimization: Analyze optimal exit times vs fixed 3:30 PM
3. Comparison Framework: Original vs Enhanced performance

Key Enhancements:
- Volume confirmation using first 5-10 minutes of trading
- Dynamic exit timing based on statistical analysis
- Fade vs Follow logic based on volume conviction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedGapFadeAnalyzer:
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
        
        csv_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found for {symbol} in {file_path}")
        
        csv_file = csv_files[0]
        
        df = pd.read_csv(os.path.join(file_path, csv_file))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df['trading_day'] = df['Date'].dt.date
        df['time'] = df['Date'].dt.time
        
        return df
    
    def identify_market_open_with_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced market open identification with volume analysis
        Returns market open data with first 5-10 minutes volume stats
        """
        # Filter for regular trading hours
        market_start = time(9, 30)
        market_end = time(16, 0)
        
        market_hours = df[
            (df['time'] >= market_start) & 
            (df['time'] <= market_end)
        ].copy()
        
        if market_hours.empty:
            print(f"Warning: No market hours data found. Using all available data.")
            market_hours = df.copy()
        
        # Group by trading day and get comprehensive opening data
        daily_opens = []
        
        for trading_day, day_data in market_hours.groupby('trading_day'):
            day_data = day_data.sort_values('Date').reset_index(drop=True)
            
            if day_data.empty:
                continue
            
            # Market open bar (first bar of the day)
            open_bar = day_data.iloc[0]
            
            # Get first 10 minutes of data (2 bars for 5min timeframe)
            first_10_min = day_data.head(2)
            
            # Get first 5 minutes volume (opening bar)
            first_5min_volume = open_bar['Volume']
            
            # Get first 10 minutes volume (sum of first 2 bars)
            first_10min_volume = first_10_min['Volume'].sum()
            
            # Calculate average volume for the first hour (12 bars for 5min)
            first_hour = day_data.head(12)
            avg_first_hour_volume = first_hour['Volume'].mean()
            
            daily_opens.append({
                'trading_day': trading_day,
                'Date': open_bar['Date'],
                'Open': open_bar['Open'],
                'High': open_bar['High'],
                'Low': open_bar['Low'],
                'Close': open_bar['Close'],
                'Volume': open_bar['Volume'],
                'first_5min_volume': first_5min_volume,
                'first_10min_volume': first_10min_volume,
                'avg_first_hour_volume': avg_first_hour_volume,
                'time': open_bar['time']
            })
        
        return pd.DataFrame(daily_opens)
    
    def calculate_enhanced_daily_stats(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """
        Calculate enhanced daily statistics with volume analysis
        """
        market_opens = self.identify_market_open_with_volume(df)
        market_opens = market_opens.sort_values('Date').reset_index(drop=True)
        
        # Previous day's close for gap calculation
        market_opens['prev_close'] = market_opens['Close'].shift(1)
        
        # Gap percentage
        market_opens['gap_percent'] = (market_opens['Open'] - market_opens['prev_close']) / market_opens['prev_close']
        
        # Rolling volume statistics (avoid lookahead)
        market_opens['avg_20d_volume'] = market_opens['Volume'].rolling(
            window=lookback, min_periods=1
        ).mean().shift(1)
        
        market_opens['avg_20d_5min_volume'] = market_opens['first_5min_volume'].rolling(
            window=lookback, min_periods=1
        ).mean().shift(1)
        
        market_opens['avg_20d_10min_volume'] = market_opens['first_10min_volume'].rolling(
            window=lookback, min_periods=1
        ).mean().shift(1)
        
        # Volume ratios
        market_opens['opening_volume_ratio'] = market_opens['Volume'] / market_opens['avg_20d_volume']
        market_opens['first_5min_ratio'] = market_opens['first_5min_volume'] / market_opens['avg_20d_5min_volume']
        market_opens['first_10min_ratio'] = market_opens['first_10min_volume'] / market_opens['avg_20d_10min_volume']
        
        # Remove rows with missing data
        market_opens = market_opens.dropna(subset=['prev_close', 'avg_20d_volume'])
        
        return market_opens
    
    def identify_enhanced_opportunities(self, daily_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced opportunity identification with volume-based fade/follow logic
        """
        opportunities = daily_stats.copy()
        
        # Gap criteria: 5-15% gap
        gap_criteria = (np.abs(opportunities['gap_percent']) >= 0.05) & (np.abs(opportunities['gap_percent']) <= 0.15)
        
        # Enhanced volume logic from your README
        # FADE: Low volume (< 3x average) means no conviction - fade the gap
        fade_volume_criteria = opportunities['first_5min_ratio'] < 3.0
        
        # FOLLOW: High volume (> 10x average) means real move - follow the gap  
        follow_volume_criteria = opportunities['first_5min_ratio'] > 10.0
        
        # Identify fade opportunities
        fade_opportunities = opportunities[gap_criteria & fade_volume_criteria].copy()
        fade_opportunities['strategy_type'] = 'fade'
        fade_opportunities['trade_direction'] = np.where(
            fade_opportunities['gap_percent'] > 0, 'short', 'long'
        )
        
        # Identify follow opportunities
        follow_opportunities = opportunities[gap_criteria & follow_volume_criteria].copy()
        follow_opportunities['strategy_type'] = 'follow'
        follow_opportunities['trade_direction'] = np.where(
            follow_opportunities['gap_percent'] > 0, 'long', 'short'  # Follow the gap direction
        )
        
        # Combine both strategies
        all_opportunities = pd.concat([fade_opportunities, follow_opportunities], ignore_index=True)
        
        return all_opportunities
    
    def analyze_exit_timing(self, opportunities: pd.DataFrame, full_df: pd.DataFrame) -> Dict:
        """
        Analyze optimal exit times by testing different time stops
        """
        time_stops = [
            time(10, 30),  # 1 hour
            time(11, 30),  # 2 hours  
            time(12, 30),  # 3 hours
            time(13, 30),  # 4 hours
            time(14, 30),  # 5 hours
            time(15, 30),  # 6 hours (original)
            time(16, 0),   # Market close
        ]
        
        timing_results = {}
        
        for stop_time in time_stops:
            print(f"Testing exit time: {stop_time}")
            trades = self.simulate_enhanced_trades(opportunities, full_df, time_stop=stop_time)
            
            if not trades.empty:
                perf = self.calculate_performance_metrics(trades)
                timing_results[str(stop_time)] = {
                    'performance': perf,
                    'trades': trades
                }
        
        return timing_results
    
    def simulate_enhanced_trades(self, opportunities: pd.DataFrame, full_df: pd.DataFrame, 
                                time_stop: time = time(15, 30)) -> pd.DataFrame:
        """
        Enhanced trade simulation with fade/follow logic and flexible exit timing
        """
        trades = []
        
        for idx, opp in opportunities.iterrows():
            trade_date = opp['trading_day']
            entry_price = opp['Open']
            direction = opp['trade_direction']
            strategy_type = opp['strategy_type']
            
            # Set targets based on strategy type
            if strategy_type == 'fade':
                target_price = opp['prev_close']  # Gap fill for fade
            else:  # follow
                # For follow strategy, target is continuation of gap
                gap_size = abs(opp['gap_percent'])
                if direction == 'long':
                    target_price = entry_price * (1 + gap_size * 0.5)  # 50% more gap extension
                else:  # short
                    target_price = entry_price * (1 - gap_size * 0.5)
            
            # Stop loss (2% beyond entry)
            if direction == 'short':
                stop_price = entry_price * 1.02
            else:  # long
                stop_price = entry_price * 0.98
            
            # Get intraday data
            day_data = full_df[full_df['trading_day'] == trade_date].copy()
            day_data = day_data.sort_values('Date')
            
            if day_data.empty:
                continue
            
            entry_time = day_data.iloc[0]['Date']
            exit_time = None
            exit_price = None
            exit_reason = None
            
            # Simulate trade execution
            for i, row in day_data.iterrows():
                current_time = row['Date']
                current_price = row['Close']
                
                if current_time == entry_time:
                    continue
                
                # Time stop check
                if current_time.time() >= time_stop:
                    exit_time = current_time
                    exit_price = current_price
                    exit_reason = 'time_stop'
                    break
                
                # Stop loss and target checks
                if direction == 'short':
                    if row['High'] >= stop_price:
                        exit_time = current_time
                        exit_price = stop_price
                        exit_reason = 'stop_loss'
                        break
                    elif row['Low'] <= target_price:
                        exit_time = current_time
                        exit_price = target_price
                        exit_reason = 'target'
                        break
                else:  # long
                    if row['Low'] <= stop_price:
                        exit_time = current_time
                        exit_price = stop_price
                        exit_reason = 'stop_loss'
                        break
                    elif row['High'] >= target_price:
                        exit_time = current_time
                        exit_price = target_price
                        exit_reason = 'target'
                        break
            
            # Handle case where no exit condition met
            if exit_time is None:
                last_bar = day_data.iloc[-1]
                exit_time = last_bar['Date']
                exit_price = last_bar['Close']
                exit_reason = 'eod'
            
            # Calculate P&L
            if direction == 'short':
                pnl = entry_price - exit_price
                pnl_pct = pnl / entry_price
            else:  # long
                pnl = exit_price - entry_price
                pnl_pct = pnl / entry_price
            
            trades.append({
                'symbol': opp.get('symbol', 'unknown'),
                'date': trade_date,
                'strategy_type': strategy_type,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'target_price': target_price,
                'stop_price': stop_price,
                'gap_percent': opp['gap_percent'],
                'first_5min_ratio': opp['first_5min_ratio'],
                'first_10min_ratio': opp['first_10min_ratio'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'winner': pnl > 0
            })
        
        return pd.DataFrame(trades)
    
    def calculate_performance_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        if trades.empty:
            return {'error': 'No trades to analyze'}
        
        total_trades = len(trades)
        winners = trades['winner'].sum()
        losers = total_trades - winners
        win_rate = winners / total_trades if total_trades > 0 else 0
        
        total_pnl_pct = trades['pnl_pct'].sum()
        avg_win = trades[trades['winner']]['pnl_pct'].mean() if winners > 0 else 0
        avg_loss = trades[~trades['winner']]['pnl_pct'].mean() if losers > 0 else 0
        
        reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        max_win = trades['pnl_pct'].max()
        max_loss = trades['pnl_pct'].min()
        
        exit_reasons = trades['exit_reason'].value_counts().to_dict()
        
        # Strategy type breakdown
        strategy_breakdown = {}
        for strategy in trades['strategy_type'].unique():
            strategy_trades = trades[trades['strategy_type'] == strategy]
            if not strategy_trades.empty:
                strategy_breakdown[strategy] = {
                    'trades': len(strategy_trades),
                    'win_rate': strategy_trades['winner'].mean(),
                    'total_pnl_pct': strategy_trades['pnl_pct'].sum(),
                    'avg_trade_pct': strategy_trades['pnl_pct'].mean()
                }
        
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
            'exit_reasons': exit_reasons,
            'strategy_breakdown': strategy_breakdown
        }
    
    def analyze_symbol_enhanced(self, symbol: str, timeframe: str = "5m") -> Dict:
        """Enhanced analysis for a single symbol"""
        print(f"\nEnhanced analysis for {symbol}...")
        
        try:
            # Load data
            df = self.load_data(symbol, timeframe)
            print(f"Loaded {len(df)} bars")
            
            # Enhanced daily statistics
            daily_stats = self.calculate_enhanced_daily_stats(df)
            print(f"Generated enhanced stats for {len(daily_stats)} trading days")
            
            # Enhanced opportunities (fade + follow)
            opportunities = self.identify_enhanced_opportunities(daily_stats)
            print(f"Found {len(opportunities)} enhanced opportunities")
            
            if opportunities.empty:
                return {
                    'symbol': symbol,
                    'total_opportunities': 0,
                    'total_trades': 0,
                    'trades': pd.DataFrame()
                }
            
            # Add symbol to opportunities for tracking
            opportunities['symbol'] = symbol
            
            # Analyze optimal exit timing
            print("Analyzing exit timing optimization...")
            timing_results = self.analyze_exit_timing(opportunities, df)
            
            # Use best performing time stop for final results
            best_time_stop = max(timing_results.keys(), 
                               key=lambda x: timing_results[x]['performance'].get('reward_risk_ratio', 0))
            
            print(f"Best exit time: {best_time_stop}")
            
            best_trades = timing_results[best_time_stop]['trades']
            
            return {
                'symbol': symbol,
                'total_opportunities': len(opportunities),
                'total_trades': len(best_trades),
                'trades': best_trades,
                'opportunities': opportunities,
                'timing_results': timing_results,
                'best_exit_time': best_time_stop
            }
            
        except Exception as e:
            print(f"Error in enhanced analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'total_trades': 0,
                'trades': pd.DataFrame()
            }
    
    def run_enhanced_analysis(self, timeframe: str = "5m"):
        """Run complete enhanced analysis"""
        print(f"=== Enhanced Gap Fade Strategy Analysis ===")
        print(f"Version 2: Enhanced Volume + Exit Timing Optimization")
        print(f"Symbols: {self.symbols}")
        print(f"Timeframe: {timeframe}")
        
        all_trades = []
        symbol_results = {}
        
        for symbol in self.symbols:
            result = self.analyze_symbol_enhanced(symbol, timeframe)
            symbol_results[symbol] = result
            
            if not result['trades'].empty:
                all_trades.append(result['trades'])
        
        # Combine results
        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)
        else:
            combined_trades = pd.DataFrame()
        
        # Calculate performance
        overall_performance = self.calculate_performance_metrics(combined_trades)
        
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
        
        self.print_enhanced_summary()
        
        return self.results
    
    def print_enhanced_summary(self):
        """Print enhanced analysis summary"""
        print(f"\n{'='*60}")
        print(f"ENHANCED STRATEGY PERFORMANCE")
        print(f"{'='*60}")
        
        perf = self.results['overall_performance']
        
        if 'error' in perf:
            print("No trades found")
            return
        
        print(f"Total Trades: {perf['total_trades']}")
        print(f"Win Rate: {perf['win_rate']:.1%}")
        print(f"Total P&L: {perf['total_pnl_pct']:.2%}")
        print(f"Average Win: {perf['avg_win_pct']:.2%}")
        print(f"Average Loss: {perf['avg_loss_pct']:.2%}")
        print(f"Reward/Risk Ratio: {perf['reward_risk_ratio']:.2f}")
        print(f"Best Trade: {perf['max_win_pct']:.2%}")
        print(f"Worst Trade: {perf['max_loss_pct']:.2%}")
        
        # Strategy breakdown
        if 'strategy_breakdown' in perf:
            print(f"\nStrategy Breakdown:")
            for strategy, stats in perf['strategy_breakdown'].items():
                print(f"  {strategy.upper()}:")
                print(f"    Trades: {stats['trades']}")
                print(f"    Win Rate: {stats['win_rate']:.1%}")
                print(f"    Total P&L: {stats['total_pnl_pct']:.2%}")
                print(f"    Avg per Trade: {stats['avg_trade_pct']:.2%}")
        
        print(f"\nExit Reasons:")
        for reason, count in perf['exit_reasons'].items():
            print(f"  {reason}: {count} ({count/perf['total_trades']:.1%})")
        
        # Best exit times per symbol
        print(f"\n{'='*60}")
        print(f"OPTIMAL EXIT TIMES BY SYMBOL")
        print(f"{'='*60}")
        
        for symbol, result in self.results['symbol_results'].items():
            if 'best_exit_time' in result:
                print(f"{symbol}: {result['best_exit_time']}")

def main():
    symbols = ["ISSC", "CCLD", "PLX", "LFVN", "BDTX"]
    
    analyzer = EnhancedGapFadeAnalyzer(symbols)
    results = analyzer.run_enhanced_analysis(timeframe="5m")
    
    # Save results
    if not results['combined_trades'].empty:
        results['combined_trades'].to_csv("gap_fade_enhanced_results.csv", index=False)
        print(f"\nEnhanced results saved to gap_fade_enhanced_results.csv")

if __name__ == "__main__":
    main()