"""
Enhanced Gap Fade Strategy with 8-K Filings Integration

This builds on the successful gap fade strategy by integrating SEC 8-K filings analysis
to potentially improve trade selection and timing. The hypothesis is that:

1. Gaps near 8-K filing dates may behave differently (more/less likely to fill)
2. Different 8-K item types may correlate with gap behavior
3. Filing timing relative to gaps can provide additional signal

Key Enhancements:
- 8-K filing proximity analysis
- Item-specific categorization (earnings, material events, etc.)
- Filing-gap correlation scoring
- Enhanced trade selection criteria
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multi_data_sources.combiners.sec_filings_combiner import SECFilingsCombiner


class GapFade8KAnalyzer:
    def __init__(self, symbols: List[str], data_path: str = "storage/data/alpaca"):
        self.symbols = symbols
        self.data_path = data_path
        self.results = {}
        self.combiner = SECFilingsCombiner()
        
        # 8-K Item impact categories (from our combiner)
        self.high_impact_items = [
            '1.01',  # Entry into Material Agreement
            '2.02',  # Results of Operations and Financial Condition (earnings)
            '5.02',  # Departure of Directors or Principal Officers
            '5.03',  # Amendments to Articles of Incorporation or Bylaws
            '7.01',  # Regulation FD Disclosure
            '8.01'   # Other Events
        ]
        
    def load_combined_data(self, symbol: str, timeframe: str = "5m", 
                          date_range: Dict[str, str] = None) -> pd.DataFrame:
        """
        Load market data combined with 8-K filings features
        """
        if date_range is None:
            date_range = {
                'start': '2016-01-01',
                'end': '2025-08-25'
            }
        
        try:
            # Use our SEC filings combiner to get combined dataset
            combined_df = self.combiner.combine_data(
                symbol=symbol,
                timeframe=timeframe,
                date_range=date_range,
                data_sources=['sec_filings']
            )
            
            # Add time and trading day columns for analysis
            combined_df.reset_index(inplace=True)
            combined_df['time'] = combined_df['Date'].dt.time
            combined_df['trading_day'] = combined_df['Date'].dt.date
            
            return combined_df
            
        except Exception as e:
            print(f"Error loading combined data for {symbol}: {e}")
            return pd.DataFrame()
    
    def identify_market_open_with_filings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced market open identification with 8-K filings data
        """
        if df.empty:
            return pd.DataFrame()
        
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
        
        # Get market opens with all 8-K features
        market_opens = market_hours.groupby('trading_day').first().reset_index()
        
        return market_opens
    
    def calculate_enhanced_daily_stats_8k(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """
        Calculate daily statistics with 8-K filings integration
        """
        market_opens = self.identify_market_open_with_filings(df)
        if market_opens.empty:
            return pd.DataFrame()
            
        market_opens = market_opens.sort_values('Date').reset_index(drop=True)
        
        # Basic gap calculations
        market_opens['prev_close'] = market_opens['Close'].shift(1)
        market_opens['gap_percent'] = (market_opens['Open'] - market_opens['prev_close']) / market_opens['prev_close']
        
        # Volume statistics
        market_opens['avg_20d_volume'] = market_opens['Volume'].rolling(
            window=lookback, min_periods=1
        ).mean().shift(1)
        market_opens['volume_ratio'] = market_opens['Volume'] / market_opens['avg_20d_volume']
        
        # 8-K Enhanced Features Analysis
        # Create filing impact score based on recent filings
        market_opens['filing_impact_score'] = (
            # Recent high-impact filings get higher weight
            (market_opens['has_recent_high_impact'] * 3) +
            # Recent 8-K filings
            (market_opens['has_recent_8k'] * 2) +
            # Recent material agreements
            (market_opens.get('has_recent_item_1_01', 0) * 2) +
            # Recent earnings-related filings
            (market_opens.get('has_recent_item_2_02', 0) * 3) +
            # Recent management changes
            (market_opens.get('has_recent_item_5_02', 0) * 2)
        )
        
        # Filing proximity categories
        market_opens['filing_proximity_category'] = 'none'
        market_opens.loc[market_opens['days_since_last_filing'] <= 1, 'filing_proximity_category'] = 'same_day'
        market_opens.loc[
            (market_opens['days_since_last_filing'] > 1) & 
            (market_opens['days_since_last_filing'] <= 3), 'filing_proximity_category'
        ] = 'within_3_days'
        market_opens.loc[
            (market_opens['days_since_last_filing'] > 3) & 
            (market_opens['days_since_last_filing'] <= 7), 'filing_proximity_category'
        ] = 'within_week'
        
        # Filing frequency signal
        market_opens['high_filing_activity'] = (market_opens['filings_in_last_30_days'] >= 3).astype(int)
        
        # Remove rows with missing essential data
        market_opens = market_opens.dropna(subset=['prev_close', 'avg_20d_volume'])
        
        return market_opens
    
    def identify_enhanced_8k_opportunities(self, daily_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced opportunity identification using 8-K filings analysis
        """
        if daily_stats.empty:
            return pd.DataFrame()
            
        opportunities = daily_stats.copy()
        
        # Basic gap criteria (5-15% gap)
        gap_criteria = (np.abs(opportunities['gap_percent']) >= 0.05) & (np.abs(opportunities['gap_percent']) <= 0.15)
        
        # Enhanced volume criteria with 8-K context
        # If there's recent high-impact filing activity, adjust volume thresholds
        base_volume_threshold = 2.0
        
        # Adjust volume threshold based on filing activity
        volume_threshold = np.where(
            opportunities['filing_impact_score'] > 2,
            base_volume_threshold * 1.5,  # Higher threshold when recent filings (more volume expected)
            base_volume_threshold
        )
        
        volume_criteria = opportunities['volume_ratio'] < volume_threshold
        
        # 8-K Enhanced Filtering
        # Strategy 1: Standard fade (low volume, no recent filings)
        standard_fade = gap_criteria & volume_criteria & (opportunities['filing_impact_score'] == 0)
        
        # Strategy 2: Filing-influenced fade (recent filings but still low volume)
        filing_fade = gap_criteria & volume_criteria & (opportunities['filing_impact_score'] > 0) & (opportunities['filing_impact_score'] <= 5)
        
        # Strategy 3: High-impact filing fade (recent high-impact filings with controlled volume)
        high_impact_fade = gap_criteria & (opportunities['volume_ratio'] < base_volume_threshold * 2) & (opportunities['filing_impact_score'] > 5)
        
        # Strategy 4: Earnings-related fade (recent 2.02 items)
        earnings_fade = (
            gap_criteria & 
            volume_criteria & 
            (opportunities.get('has_recent_item_2_02', 0) == 1)
        )
        
        # Combine strategies
        all_opportunities = []
        
        # Standard fade opportunities
        standard_opps = opportunities[standard_fade].copy()
        if not standard_opps.empty:
            standard_opps['strategy_type'] = 'standard_fade'
            standard_opps['trade_direction'] = np.where(standard_opps['gap_percent'] > 0, 'short', 'long')
            all_opportunities.append(standard_opps)
        
        # Filing-influenced fade opportunities  
        filing_opps = opportunities[filing_fade].copy()
        if not filing_opps.empty:
            filing_opps['strategy_type'] = 'filing_fade'
            filing_opps['trade_direction'] = np.where(filing_opps['gap_percent'] > 0, 'short', 'long')
            all_opportunities.append(filing_opps)
        
        # High-impact filing fade opportunities
        high_impact_opps = opportunities[high_impact_fade].copy()
        if not high_impact_opps.empty:
            high_impact_opps['strategy_type'] = 'high_impact_fade'
            high_impact_opps['trade_direction'] = np.where(high_impact_opps['gap_percent'] > 0, 'short', 'long')
            all_opportunities.append(high_impact_opps)
        
        # Earnings fade opportunities
        earnings_opps = opportunities[earnings_fade].copy()
        if not earnings_opps.empty:
            earnings_opps['strategy_type'] = 'earnings_fade'
            earnings_opps['trade_direction'] = np.where(earnings_opps['gap_percent'] > 0, 'short', 'long')
            all_opportunities.append(earnings_opps)
        
        if all_opportunities:
            return pd.concat(all_opportunities, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def simulate_8k_enhanced_trades(self, opportunities: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate trades with 8-K enhanced logic
        """
        if opportunities.empty or full_df.empty:
            return pd.DataFrame()
            
        trades = []
        
        for idx, opp in opportunities.iterrows():
            trade_date = opp['trading_day']
            entry_price = opp['Open']
            direction = opp['trade_direction']
            strategy_type = opp['strategy_type']
            
            # Enhanced target and stop logic based on 8-K analysis
            if strategy_type in ['standard_fade', 'filing_fade']:
                target_price = opp['prev_close']  # Standard gap fill
                stop_multiplier = 0.02  # 2% stop
            elif strategy_type == 'high_impact_fade':
                # More conservative targets for high-impact filings (partial gap fill)
                gap_size = abs(opp['gap_percent'])
                if direction == 'short':
                    target_price = entry_price - (entry_price * gap_size * 0.7)  # 70% gap fill
                else:  # long
                    target_price = entry_price + (entry_price * gap_size * 0.7)
                stop_multiplier = 0.015  # Tighter 1.5% stop for high-impact
            elif strategy_type == 'earnings_fade':
                # Earnings-related gaps may fill more completely
                target_price = opp['prev_close']
                stop_multiplier = 0.025  # Slightly wider 2.5% stop for earnings volatility
            else:
                target_price = opp['prev_close']
                stop_multiplier = 0.02
            
            # Calculate stop loss
            if direction == 'short':
                stop_price = entry_price * (1 + stop_multiplier)
            else:  # long
                stop_price = entry_price * (1 - stop_multiplier)
            
            # Get intraday data for this trading day
            day_data = full_df[full_df['trading_day'] == trade_date].copy()
            day_data = day_data.sort_values('Date')
            
            if day_data.empty:
                continue
            
            entry_time = day_data.iloc[0]['Date']
            
            # Enhanced exit timing based on 8-K context
            # More aggressive exits for filing-related trades
            if strategy_type in ['high_impact_fade', 'earnings_fade']:
                time_stop = time(14, 30)  # Earlier exit for filing-related trades
            else:
                time_stop = time(15, 30)  # Standard exit time
            
            exit_time = None
            exit_price = None
            exit_reason = None
            
            # Simulate trade execution
            for i, row in day_data.iterrows():
                current_time = row['Date']
                
                if current_time == entry_time:
                    continue
                
                # Time stop check
                if current_time.time() >= time_stop:
                    exit_time = current_time
                    exit_price = row['Close']
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
            
            # Handle end of day
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
            
            # Collect 8-K specific data for analysis
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
                'volume_ratio': opp['volume_ratio'],
                'filing_impact_score': opp['filing_impact_score'],
                'filing_proximity_category': opp['filing_proximity_category'],
                'days_since_last_filing': opp['days_since_last_filing'],
                'filings_in_last_30_days': opp['filings_in_last_30_days'],
                'has_recent_high_impact': opp['has_recent_high_impact'],
                'has_recent_8k': opp['has_recent_8k'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'winner': pnl > 0
            })
        
        return pd.DataFrame(trades)
    
    def calculate_8k_performance_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate performance metrics with 8-K analysis"""
        if trades.empty:
            return {'error': 'No trades to analyze'}
        
        # Basic performance metrics
        total_trades = len(trades)
        winners = trades['winner'].sum()
        win_rate = winners / total_trades if total_trades > 0 else 0
        
        total_pnl_pct = trades['pnl_pct'].sum()
        avg_win = trades[trades['winner']]['pnl_pct'].mean() if winners > 0 else 0
        avg_loss = trades[~trades['winner']]['pnl_pct'].mean() if (total_trades - winners) > 0 else 0
        
        reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Strategy type breakdown
        strategy_breakdown = {}
        for strategy in trades['strategy_type'].unique():
            strategy_trades = trades[trades['strategy_type'] == strategy]
            strategy_breakdown[strategy] = {
                'trades': len(strategy_trades),
                'win_rate': strategy_trades['winner'].mean(),
                'total_pnl_pct': strategy_trades['pnl_pct'].sum(),
                'avg_trade_pct': strategy_trades['pnl_pct'].mean()
            }
        
        # Filing proximity analysis
        proximity_analysis = {}
        for proximity in trades['filing_proximity_category'].unique():
            prox_trades = trades[trades['filing_proximity_category'] == proximity]
            proximity_analysis[proximity] = {
                'trades': len(prox_trades),
                'win_rate': prox_trades['winner'].mean(),
                'avg_pnl_pct': prox_trades['pnl_pct'].mean()
            }
        
        # Filing impact score correlation
        impact_correlation = trades['filing_impact_score'].corr(trades['pnl_pct']) if len(trades) > 1 else 0
        
        return {
            'total_trades': total_trades,
            'winners': winners,
            'win_rate': win_rate,
            'total_pnl_pct': total_pnl_pct,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'reward_risk_ratio': reward_risk_ratio,
            'strategy_breakdown': strategy_breakdown,
            'proximity_analysis': proximity_analysis,
            'filing_impact_correlation': impact_correlation,
            'exit_reasons': trades['exit_reason'].value_counts().to_dict()
        }
    
    def analyze_symbol_with_8k(self, symbol: str, timeframe: str = "5m") -> Dict:
        """Analyze a single symbol with 8-K integration"""
        print(f"\n8-K Enhanced Analysis for {symbol}...")
        
        try:
            # Load combined market + 8-K data
            combined_df = self.load_combined_data(symbol, timeframe)
            if combined_df.empty:
                return {'symbol': symbol, 'error': 'No combined data available', 'total_trades': 0}
            
            print(f"Loaded {len(combined_df)} bars with 8-K features")
            
            # Calculate enhanced daily statistics
            daily_stats = self.calculate_enhanced_daily_stats_8k(combined_df)
            if daily_stats.empty:
                return {'symbol': symbol, 'error': 'No daily stats generated', 'total_trades': 0}
            
            print(f"Generated enhanced stats for {len(daily_stats)} trading days")
            
            # Identify 8-K enhanced opportunities
            opportunities = self.identify_enhanced_8k_opportunities(daily_stats)
            print(f"Found {len(opportunities)} 8-K enhanced opportunities")
            
            if opportunities.empty:
                return {
                    'symbol': symbol,
                    'total_opportunities': 0,
                    'total_trades': 0,
                    'trades': pd.DataFrame()
                }
            
            # Add symbol tracking
            opportunities['symbol'] = symbol
            
            # Simulate trades
            trades = self.simulate_8k_enhanced_trades(opportunities, combined_df)
            print(f"Simulated {len(trades)} 8-K enhanced trades")
            
            return {
                'symbol': symbol,
                'total_opportunities': len(opportunities),
                'total_trades': len(trades),
                'trades': trades,
                'opportunities': opportunities,
                'daily_stats': daily_stats
            }
            
        except Exception as e:
            print(f"Error analyzing {symbol} with 8-K: {e}")
            import traceback
            traceback.print_exc()
            return {
                'symbol': symbol,
                'error': str(e),
                'total_trades': 0,
                'trades': pd.DataFrame()
            }
    
    def run_8k_analysis(self, timeframe: str = "5m"):
        """Run complete 8-K enhanced analysis"""
        print(f"=== Gap Fade Strategy with 8-K Filings Analysis ===")
        print(f"Enhanced with SEC 8-K filing proximity and impact analysis")
        print(f"Symbols: {self.symbols}")
        print(f"Timeframe: {timeframe}")
        
        all_trades = []
        symbol_results = {}
        
        for symbol in self.symbols:
            result = self.analyze_symbol_with_8k(symbol, timeframe)
            symbol_results[symbol] = result
            
            if 'trades' in result and not result['trades'].empty:
                all_trades.append(result['trades'])
        
        # Combine results
        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)
        else:
            combined_trades = pd.DataFrame()
        
        # Calculate performance
        overall_performance = self.calculate_8k_performance_metrics(combined_trades)
        
        symbol_performance = {}
        for symbol, result in symbol_results.items():
            if 'trades' in result and not result['trades'].empty:
                symbol_performance[symbol] = self.calculate_8k_performance_metrics(result['trades'])
        
        self.results = {
            'combined_trades': combined_trades,
            'overall_performance': overall_performance,
            'symbol_performance': symbol_performance,
            'symbol_results': symbol_results
        }
        
        self.print_8k_summary()
        return self.results
    
    def print_8k_summary(self):
        """Print 8-K enhanced analysis summary"""
        print(f"\n{'='*70}")
        print(f"8-K ENHANCED STRATEGY PERFORMANCE")
        print(f"{'='*70}")
        
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
        
        # 8-K Strategy breakdown
        print(f"\n8-K Enhanced Strategy Breakdown:")
        for strategy, stats in perf['strategy_breakdown'].items():
            print(f"  {strategy.upper()}:")
            print(f"    Trades: {stats['trades']}")
            print(f"    Win Rate: {stats['win_rate']:.1%}")
            print(f"    Total P&L: {stats['total_pnl_pct']:.2%}")
            print(f"    Avg per Trade: {stats['avg_trade_pct']:.2%}")
        
        # Filing proximity analysis
        print(f"\nFiling Proximity Analysis:")
        for proximity, stats in perf['proximity_analysis'].items():
            print(f"  {proximity}: {stats['trades']} trades, {stats['win_rate']:.1%} win rate, {stats['avg_pnl_pct']:.2%} avg P&L")
        
        print(f"\nFiling Impact Score Correlation with P&L: {perf['filing_impact_correlation']:.3f}")
        
        print(f"\nExit Reasons:")
        for reason, count in perf['exit_reasons'].items():
            print(f"  {reason}: {count} ({count/perf['total_trades']:.1%})")
    
    def save_8k_results(self, filename: str = "gap_fade_8k_enhanced_results.csv"):
        """Save 8-K enhanced results"""
        if not self.results['combined_trades'].empty:
            self.results['combined_trades'].to_csv(filename, index=False)
            print(f"\n8-K enhanced results saved to {filename}")


def main():
    # Test with BDTX first since we have 8-K data for it
    symbols = ["BDTX"]  # Start with BDTX which has 8-K data
    
    analyzer = GapFade8KAnalyzer(symbols)
    results = analyzer.run_8k_analysis(timeframe="5m")
    
    # Save results
    analyzer.save_8k_results()
    
    print(f"\n{'='*70}")
    print(f"8-K ENHANCED STRATEGY VALIDATION COMPLETE")
    print(f"{'='*70}")
    print("Key Insights:")
    print("• This tests if 8-K filings can enhance gap fade strategy performance")
    print("• Analyzes filing proximity, impact scores, and item-specific patterns")
    print("• Compares different strategy variants (standard, filing-based, high-impact)")
    print("• Shows correlation between filing activity and gap behavior")
    print("• Ready for expansion to other symbols with 8-K data")


if __name__ == "__main__":
    main()