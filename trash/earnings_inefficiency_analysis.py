#!/usr/bin/env python3
"""
Comprehensive earnings inefficiency analysis using yfinance.

This script demonstrates how to find market inefficiencies around earnings events
by combining earnings announcement data with OHLCV price data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class EarningsInefficiencyAnalyzer:
    """Analyze market inefficiencies around earnings announcements."""
    
    def __init__(self, symbols: List[str]):
        """Initialize with list of symbols to analyze."""
        self.symbols = symbols
        self.earnings_data = {}
        self.price_data = {}
        self.analysis_results = {}
    
    def collect_earnings_data(self) -> Dict:
        """Collect comprehensive earnings data for all symbols."""
        print("🔍 COLLECTING EARNINGS DATA")
        print("=" * 40)
        
        for symbol in self.symbols:
            print(f"📊 Processing {symbol}...")
            
            ticker = yf.Ticker(symbol)
            symbol_data = {}
            
            try:
                # Get historical earnings dates with surprises
                earnings_dates = ticker.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    symbol_data['earnings_dates'] = earnings_dates
                    symbol_data['surprise_stats'] = self._calculate_surprise_stats(earnings_dates)
                
                # Get upcoming earnings
                calendar = ticker.calendar
                if calendar is not None:
                    symbol_data['upcoming_earnings'] = calendar
                
                # Get analyst targets
                targets = ticker.analyst_price_targets
                if targets is not None:
                    symbol_data['analyst_targets'] = targets
                
                self.earnings_data[symbol] = symbol_data
                print(f"   ✅ Collected earnings data for {symbol}")
                
            except Exception as e:
                print(f"   ❌ Error collecting data for {symbol}: {e}")
                self.earnings_data[symbol] = {}
        
        return self.earnings_data
    
    def collect_price_data(self, period: str = "3y") -> Dict:
        """Collect OHLCV price data for all symbols."""
        print(f"\n💲 COLLECTING PRICE DATA ({period})")
        print("=" * 40)
        
        for symbol in self.symbols:
            print(f"📈 Downloading {symbol} price data...")
            
            try:
                ticker = yf.Ticker(symbol)
                price_data = ticker.history(period=period)
                
                if not price_data.empty:
                    # Add technical indicators
                    price_data = self._add_technical_indicators(price_data)
                    self.price_data[symbol] = price_data
                    print(f"   ✅ {len(price_data)} days of price data")
                else:
                    print(f"   ❌ No price data available")
                    
            except Exception as e:
                print(f"   ❌ Error downloading {symbol}: {e}")
        
        return self.price_data
    
    def analyze_earnings_inefficiencies(self) -> Dict:
        """Analyze various market inefficiencies around earnings."""
        print(f"\n🎯 ANALYZING EARNINGS INEFFICIENCIES")
        print("=" * 45)
        
        for symbol in self.symbols:
            if symbol not in self.earnings_data or symbol not in self.price_data:
                continue
            
            print(f"\n🔬 Analyzing {symbol} inefficiencies...")
            
            earnings_dates = self.earnings_data[symbol].get('earnings_dates')
            price_data = self.price_data[symbol]
            
            if earnings_dates is None or price_data.empty:
                continue
            
            symbol_analysis = {}
            
            # 1. Post-earnings drift analysis
            symbol_analysis['post_earnings_drift'] = self._analyze_post_earnings_drift(
                earnings_dates, price_data, symbol
            )
            
            # 2. Pre-earnings run-up analysis
            symbol_analysis['pre_earnings_runup'] = self._analyze_pre_earnings_runup(
                earnings_dates, price_data, symbol
            )
            
            # 3. Earnings surprise vs price reaction
            symbol_analysis['surprise_vs_reaction'] = self._analyze_surprise_vs_reaction(
                earnings_dates, price_data, symbol
            )
            
            # 4. Day-of-week announcement effects
            symbol_analysis['day_of_week_effects'] = self._analyze_day_of_week_effects(
                earnings_dates, price_data, symbol
            )
            
            # 5. Volatility patterns around earnings
            symbol_analysis['volatility_patterns'] = self._analyze_volatility_patterns(
                earnings_dates, price_data, symbol
            )
            
            self.analysis_results[symbol] = symbol_analysis
        
        return self.analysis_results
    
    def _calculate_surprise_stats(self, earnings_dates: pd.DataFrame) -> Dict:
        """Calculate earnings surprise statistics."""
        if 'Surprise(%)' not in earnings_dates.columns:
            return {}
        
        surprises = earnings_dates['Surprise(%)'].dropna()
        if surprises.empty:
            return {}
        
        return {
            'mean_surprise': surprises.mean(),
            'std_surprise': surprises.std(),
            'positive_surprise_rate': (surprises > 0).mean(),
            'large_surprise_rate': (abs(surprises) > 5).mean(),
            'max_positive_surprise': surprises.max(),
            'max_negative_surprise': surprises.min()
        }
    
    def _add_technical_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data."""
        df = price_data.copy()
        
        # Returns
        df['daily_return'] = df['Close'].pct_change()
        df['abs_return'] = abs(df['daily_return'])
        
        # Volatility (rolling 20-day)
        df['volatility_20d'] = df['daily_return'].rolling(20).std() * np.sqrt(252)
        
        # Moving averages
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        
        # Distance from moving averages
        df['distance_from_sma20'] = (df['Close'] - df['sma_20']) / df['sma_20']
        
        return df
    
    def _get_earnings_aligned_data(self, earnings_dates: pd.DataFrame, 
                                 price_data: pd.DataFrame, 
                                 days_before: int = 10, 
                                 days_after: int = 10) -> List[Dict]:
        """Get price data aligned with earnings dates."""
        aligned_data = []
        
        for earnings_date, row in earnings_dates.iterrows():
            # Convert earnings_date to date only (remove time component)
            earnings_date_only = earnings_date.date()
            
            # Find the closest trading day to the earnings date
            price_dates = price_data.index.date
            closest_date_idx = None
            min_diff = float('inf')
            
            for i, price_date in enumerate(price_dates):
                diff = abs((price_date - earnings_date_only).days)
                if diff < min_diff:
                    min_diff = diff
                    closest_date_idx = i
            
            if closest_date_idx is None or min_diff > 5:  # Skip if >5 days away
                continue
            
            # Get data around earnings
            start_idx = max(0, closest_date_idx - days_before)
            end_idx = min(len(price_data), closest_date_idx + days_after + 1)
            
            if end_idx - start_idx < days_before + days_after:  # Need enough data
                continue
            
            earnings_window = price_data.iloc[start_idx:end_idx].copy()
            earnings_window['days_from_earnings'] = range(-days_before, days_after + 1)[:len(earnings_window)]
            
            # Add earnings metadata
            aligned_data.append({
                'earnings_date': earnings_date,
                'surprise_pct': row.get('Surprise(%)', np.nan),
                'eps_estimate': row.get('EPS Estimate', np.nan),
                'reported_eps': row.get('Reported EPS', np.nan),
                'price_data': earnings_window
            })
        
        return aligned_data\n    \n    def _analyze_post_earnings_drift(self, earnings_dates: pd.DataFrame, \n                                   price_data: pd.DataFrame, \n                                   symbol: str) -> Dict:\n        \"\"\"Analyze post-earnings drift patterns.\"\"\"\n        print(f\"   📈 Post-earnings drift analysis...\")\n        \n        aligned_data = self._get_earnings_aligned_data(earnings_dates, price_data, 5, 20)\n        \n        if not aligned_data:\n            return {}\n        \n        drift_results = {\n            'positive_surprise_drift': [],\n            'negative_surprise_drift': [],\n            'all_earnings_drift': []\n        }\n        \n        for earnings_event in aligned_data:\n            surprise = earnings_event['surprise_pct']\n            price_window = earnings_event['price_data']\n            \n            if pd.isna(surprise):\n                continue\n            \n            # Calculate post-earnings returns (days 1-20 after earnings)\n            post_earnings_data = price_window[price_window['days_from_earnings'] > 0]\n            \n            if len(post_earnings_data) >= 10:  # Need at least 10 days\n                # Cumulative return over different periods\n                returns_3d = post_earnings_data['daily_return'].iloc[:3].sum()\n                returns_10d = post_earnings_data['daily_return'].iloc[:10].sum()\n                returns_20d = post_earnings_data['daily_return'].iloc[:min(20, len(post_earnings_data))].sum()\n                \n                drift_data = {\n                    'surprise': surprise,\n                    'returns_3d': returns_3d,\n                    'returns_10d': returns_10d,\n                    'returns_20d': returns_20d,\n                    'earnings_date': earnings_event['earnings_date']\n                }\n                \n                drift_results['all_earnings_drift'].append(drift_data)\n                \n                if surprise > 2:  # Positive surprise\n                    drift_results['positive_surprise_drift'].append(drift_data)\n                elif surprise < -2:  # Negative surprise\n                    drift_results['negative_surprise_drift'].append(drift_data)\n        \n        # Calculate summary statistics\n        summary = {}\n        for category, data in drift_results.items():\n            if data:\n                df = pd.DataFrame(data)\n                summary[category] = {\n                    'count': len(df),\n                    'mean_3d_return': df['returns_3d'].mean(),\n                    'mean_10d_return': df['returns_10d'].mean(),\n                    'mean_20d_return': df['returns_20d'].mean(),\n                    'positive_3d_rate': (df['returns_3d'] > 0).mean(),\n                    'positive_10d_rate': (df['returns_10d'] > 0).mean(),\n                    'positive_20d_rate': (df['returns_20d'] > 0).mean()\n                }\n        \n        return summary\n    \n    def _analyze_pre_earnings_runup(self, earnings_dates: pd.DataFrame, \n                                  price_data: pd.DataFrame, \n                                  symbol: str) -> Dict:\n        \"\"\"Analyze pre-earnings price run-up patterns.\"\"\"\n        print(f\"   📊 Pre-earnings run-up analysis...\")\n        \n        aligned_data = self._get_earnings_aligned_data(earnings_dates, price_data, 20, 5)\n        \n        if not aligned_data:\n            return {}\n        \n        runup_data = []\n        \n        for earnings_event in aligned_data:\n            surprise = earnings_event['surprise_pct']\n            price_window = earnings_event['price_data']\n            \n            # Calculate pre-earnings returns (days -20 to -1)\n            pre_earnings_data = price_window[price_window['days_from_earnings'] < 0]\n            \n            if len(pre_earnings_data) >= 10:\n                # Different pre-earnings periods\n                returns_20d = pre_earnings_data['daily_return'].sum()\n                returns_10d = pre_earnings_data['daily_return'].tail(10).sum()\n                returns_5d = pre_earnings_data['daily_return'].tail(5).sum()\n                \n                runup_data.append({\n                    'surprise': surprise,\n                    'returns_20d_pre': returns_20d,\n                    'returns_10d_pre': returns_10d,\n                    'returns_5d_pre': returns_5d,\n                    'earnings_date': earnings_event['earnings_date']\n                })\n        \n        if not runup_data:\n            return {}\n        \n        df = pd.DataFrame(runup_data)\n        \n        # Analyze correlation between pre-earnings runup and surprise\n        correlations = {}\n        for period in ['5d_pre', '10d_pre', '20d_pre']:\n            col = f'returns_{period}'\n            if col in df.columns and not df['surprise'].isna().all():\n                corr = df[col].corr(df['surprise'].fillna(0))\n                correlations[period] = corr\n        \n        return {\n            'count': len(df),\n            'mean_runup_5d': df['returns_5d_pre'].mean(),\n            'mean_runup_10d': df['returns_10d_pre'].mean(),\n            'mean_runup_20d': df['returns_20d_pre'].mean(),\n            'runup_surprise_correlations': correlations,\n            'positive_runup_rate_5d': (df['returns_5d_pre'] > 0).mean(),\n            'positive_runup_rate_10d': (df['returns_10d_pre'] > 0).mean()\n        }\n    \n    def _analyze_surprise_vs_reaction(self, earnings_dates: pd.DataFrame, \n                                    price_data: pd.DataFrame, \n                                    symbol: str) -> Dict:\n        \"\"\"Analyze relationship between earnings surprise and price reaction.\"\"\"\n        print(f\"   🎯 Surprise vs reaction analysis...\")\n        \n        aligned_data = self._get_earnings_aligned_data(earnings_dates, price_data, 2, 2)\n        \n        if not aligned_data:\n            return {}\n        \n        reaction_data = []\n        \n        for earnings_event in aligned_data:\n            surprise = earnings_event['surprise_pct']\n            price_window = earnings_event['price_data']\n            \n            if pd.isna(surprise):\n                continue\n            \n            # Get earnings day and next day reaction\n            earnings_day_data = price_window[price_window['days_from_earnings'] == 0]\n            next_day_data = price_window[price_window['days_from_earnings'] == 1]\n            \n            if len(earnings_day_data) > 0:\n                earnings_day_return = earnings_day_data['daily_return'].iloc[0]\n                \n                next_day_return = 0\n                if len(next_day_data) > 0:\n                    next_day_return = next_day_data['daily_return'].iloc[0]\n                \n                reaction_data.append({\n                    'surprise': surprise,\n                    'earnings_day_return': earnings_day_return,\n                    'next_day_return': next_day_return,\n                    'two_day_return': earnings_day_return + next_day_return\n                })\n        \n        if not reaction_data:\n            return {}\n        \n        df = pd.DataFrame(reaction_data)\n        \n        # Calculate correlations\n        surprise_earnings_day_corr = df['surprise'].corr(df['earnings_day_return'])\n        surprise_next_day_corr = df['surprise'].corr(df['next_day_return'])\n        surprise_two_day_corr = df['surprise'].corr(df['two_day_return'])\n        \n        # Analyze by surprise magnitude\n        large_positive_surprises = df[df['surprise'] > 5]\n        large_negative_surprises = df[df['surprise'] < -5]\n        \n        return {\n            'count': len(df),\n            'surprise_earnings_day_correlation': surprise_earnings_day_corr,\n            'surprise_next_day_correlation': surprise_next_day_corr,\n            'surprise_two_day_correlation': surprise_two_day_corr,\n            'mean_earnings_day_return': df['earnings_day_return'].mean(),\n            'mean_next_day_return': df['next_day_return'].mean(),\n            'large_positive_surprises': {\n                'count': len(large_positive_surprises),\n                'mean_earnings_day_return': large_positive_surprises['earnings_day_return'].mean() if len(large_positive_surprises) > 0 else np.nan,\n                'mean_next_day_return': large_positive_surprises['next_day_return'].mean() if len(large_positive_surprises) > 0 else np.nan\n            },\n            'large_negative_surprises': {\n                'count': len(large_negative_surprises),\n                'mean_earnings_day_return': large_negative_surprises['earnings_day_return'].mean() if len(large_negative_surprises) > 0 else np.nan,\n                'mean_next_day_return': large_negative_surprises['next_day_return'].mean() if len(large_negative_surprises) > 0 else np.nan\n            }\n        }\n    \n    def _analyze_day_of_week_effects(self, earnings_dates: pd.DataFrame, \n                                   price_data: pd.DataFrame, \n                                   symbol: str) -> Dict:\n        \"\"\"Analyze day-of-week announcement timing effects.\"\"\"\n        print(f\"   📅 Day-of-week effects analysis...\")\n        \n        aligned_data = self._get_earnings_aligned_data(earnings_dates, price_data, 2, 5)\n        \n        if not aligned_data:\n            return {}\n        \n        day_effects = {}\n        \n        for earnings_event in aligned_data:\n            earnings_date = earnings_event['earnings_date']\n            day_of_week = earnings_date.strftime('%A')\n            price_window = earnings_event['price_data']\n            surprise = earnings_event['surprise_pct']\n            \n            # Get post-earnings returns\n            post_earnings = price_window[price_window['days_from_earnings'] > 0]\n            if len(post_earnings) >= 3:\n                three_day_return = post_earnings['daily_return'].iloc[:3].sum()\n                \n                if day_of_week not in day_effects:\n                    day_effects[day_of_week] = []\n                \n                day_effects[day_of_week].append({\n                    'three_day_return': three_day_return,\n                    'surprise': surprise,\n                    'earnings_date': earnings_date\n                })\n        \n        # Calculate summary by day\n        day_summary = {}\n        for day, data in day_effects.items():\n            if data:\n                df = pd.DataFrame(data)\n                day_summary[day] = {\n                    'count': len(df),\n                    'mean_3d_return': df['three_day_return'].mean(),\n                    'median_3d_return': df['three_day_return'].median(),\n                    'positive_rate': (df['three_day_return'] > 0).mean(),\n                    'mean_surprise': df['surprise'].mean() if not df['surprise'].isna().all() else np.nan\n                }\n        \n        # Identify Tuesday/Friday pattern\n        tuesday_friday_effect = {}\n        if 'Tuesday' in day_summary and 'Friday' in day_summary:\n            tuesday_friday_effect = {\n                'tuesday_vs_friday_return_diff': day_summary['Tuesday']['mean_3d_return'] - day_summary['Friday']['mean_3d_return'],\n                'tuesday_positive_rate': day_summary['Tuesday']['positive_rate'],\n                'friday_positive_rate': day_summary['Friday']['positive_rate']\n            }\n        \n        return {\n            'day_summary': day_summary,\n            'tuesday_friday_effect': tuesday_friday_effect\n        }\n    \n    def _analyze_volatility_patterns(self, earnings_dates: pd.DataFrame, \n                                   price_data: pd.DataFrame, \n                                   symbol: str) -> Dict:\n        \"\"\"Analyze volatility patterns around earnings.\"\"\"\n        print(f\"   📊 Volatility patterns analysis...\")\n        \n        aligned_data = self._get_earnings_aligned_data(earnings_dates, price_data, 10, 10)\n        \n        if not aligned_data:\n            return {}\n        \n        volatility_data = []\n        \n        for earnings_event in aligned_data:\n            price_window = earnings_event['price_data']\n            \n            # Pre-earnings volatility (days -10 to -1)\n            pre_earnings = price_window[price_window['days_from_earnings'] < 0]\n            post_earnings = price_window[price_window['days_from_earnings'] > 0]\n            earnings_day = price_window[price_window['days_from_earnings'] == 0]\n            \n            if len(pre_earnings) >= 5 and len(post_earnings) >= 5 and len(earnings_day) > 0:\n                pre_vol = pre_earnings['abs_return'].mean()\n                post_vol = post_earnings['abs_return'].mean()\n                earnings_day_vol = earnings_day['abs_return'].iloc[0]\n                \n                volatility_data.append({\n                    'pre_earnings_volatility': pre_vol,\n                    'post_earnings_volatility': post_vol,\n                    'earnings_day_volatility': earnings_day_vol,\n                    'vol_ratio_post_pre': post_vol / pre_vol if pre_vol > 0 else np.nan,\n                    'earnings_date': earnings_event['earnings_date']\n                })\n        \n        if not volatility_data:\n            return {}\n        \n        df = pd.DataFrame(volatility_data)\n        \n        return {\n            'count': len(df),\n            'mean_pre_earnings_vol': df['pre_earnings_volatility'].mean(),\n            'mean_post_earnings_vol': df['post_earnings_volatility'].mean(),\n            'mean_earnings_day_vol': df['earnings_day_volatility'].mean(),\n            'mean_vol_ratio_post_pre': df['vol_ratio_post_pre'].mean(),\n            'volatility_increases_post_earnings_rate': (df['vol_ratio_post_pre'] > 1).mean()\n        }\n    \n    def generate_inefficiency_report(self) -> str:\n        \"\"\"Generate comprehensive inefficiency analysis report.\"\"\"\n        report = []\n        report.append(\"🎯 EARNINGS MARKET INEFFICIENCY ANALYSIS REPORT\")\n        report.append(\"=\" * 55)\n        \n        for symbol, analysis in self.analysis_results.items():\n            report.append(f\"\\n📊 {symbol} ANALYSIS RESULTS\")\n            report.append(\"-\" * 30)\n            \n            # Post-earnings drift\n            if 'post_earnings_drift' in analysis:\n                drift = analysis['post_earnings_drift']\n                report.append(f\"\\n📈 POST-EARNINGS DRIFT:\")\n                \n                if 'all_earnings_drift' in drift and drift['all_earnings_drift']:\n                    all_drift = drift['all_earnings_drift']\n                    report.append(f\"   • Sample size: {all_drift['count']} earnings events\")\n                    report.append(f\"   • Mean 3-day return: {all_drift['mean_3d_return']:.2%}\")\n                    report.append(f\"   • Mean 10-day return: {all_drift['mean_10d_return']:.2%}\")\n                    report.append(f\"   • Mean 20-day return: {all_drift['mean_20d_return']:.2%}\")\n                    \n                    if 'positive_surprise_drift' in drift and drift['positive_surprise_drift']:\n                        pos_drift = drift['positive_surprise_drift']\n                        report.append(f\"   • Positive surprise 10d drift: {pos_drift['mean_10d_return']:.2%} ({pos_drift['count']} events)\")\n            \n            # Pre-earnings runup\n            if 'pre_earnings_runup' in analysis:\n                runup = analysis['pre_earnings_runup']\n                if runup:\n                    report.append(f\"\\n📊 PRE-EARNINGS RUN-UP:\")\n                    report.append(f\"   • Mean 5-day pre-earnings return: {runup['mean_runup_5d']:.2%}\")\n                    report.append(f\"   • Mean 10-day pre-earnings return: {runup['mean_runup_10d']:.2%}\")\n                    \n                    if 'runup_surprise_correlations' in runup:\n                        corrs = runup['runup_surprise_correlations']\n                        if '5d_pre' in corrs:\n                            report.append(f\"   • 5d runup-surprise correlation: {corrs['5d_pre']:.3f}\")\n            \n            # Surprise vs reaction\n            if 'surprise_vs_reaction' in analysis:\n                surprise_reaction = analysis['surprise_vs_reaction']\n                if surprise_reaction:\n                    report.append(f\"\\n🎯 SURPRISE VS REACTION:\")\n                    report.append(f\"   • Surprise-earnings day correlation: {surprise_reaction['surprise_earnings_day_correlation']:.3f}\")\n                    report.append(f\"   • Mean earnings day return: {surprise_reaction['mean_earnings_day_return']:.2%}\")\n                    report.append(f\"   • Mean next day return: {surprise_reaction['mean_next_day_return']:.2%}\")\n            \n            # Day of week effects\n            if 'day_of_week_effects' in analysis:\n                day_effects = analysis['day_of_week_effects']\n                if 'day_summary' in day_effects:\n                    report.append(f\"\\n📅 DAY-OF-WEEK EFFECTS:\")\n                    for day, stats in day_effects['day_summary'].items():\n                        report.append(f\"   • {day}: {stats['mean_3d_return']:.2%} avg 3d return ({stats['count']} events)\")\n            \n            # Volatility patterns\n            if 'volatility_patterns' in analysis:\n                vol_patterns = analysis['volatility_patterns']\n                if vol_patterns:\n                    report.append(f\"\\n📊 VOLATILITY PATTERNS:\")\n                    report.append(f\"   • Pre-earnings volatility: {vol_patterns['mean_pre_earnings_vol']:.2%}\")\n                    report.append(f\"   • Post-earnings volatility: {vol_patterns['mean_post_earnings_vol']:.2%}\")\n                    report.append(f\"   • Earnings day volatility: {vol_patterns['mean_earnings_day_vol']:.2%}\")\n        \n        return \"\\n\".join(report)\n\ndef main():\n    \"\"\"Run comprehensive earnings inefficiency analysis.\"\"\"\n    # Test with major tech stocks\n    symbols = [\"AAPL\", \"MSFT\", \"GOOGL\", \"AMZN\", \"NVDA\"]\n    \n    analyzer = EarningsInefficiencyAnalyzer(symbols)\n    \n    # Collect data\n    analyzer.collect_earnings_data()\n    analyzer.collect_price_data()\n    \n    # Analyze inefficiencies\n    analyzer.analyze_earnings_inefficiencies()\n    \n    # Generate report\n    report = analyzer.generate_inefficiency_report()\n    print(\"\\n\" + report)\n    \n    # Save results\n    timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n    report_file = f\"/Users/dmartinsen/pers-prog/ml-pred/earnings_inefficiency_report_{timestamp}.txt\"\n    \n    with open(report_file, 'w') as f:\n        f.write(report)\n    \n    print(f\"\\n💾 Report saved to: {report_file}\")\n    \n    return analyzer\n\nif __name__ == \"__main__\":\n    analyzer = main()