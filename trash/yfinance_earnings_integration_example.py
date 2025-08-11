#!/usr/bin/env python3
"""
Practical yfinance earnings integration example for your existing earnings.py.

This shows exactly how to enhance your existing earnings component with
the proven yfinance capabilities for finding market inefficiencies.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class EnhancedEarningsAnalyzer:
    """Enhanced earnings analyzer using proven yfinance capabilities."""
    
    def __init__(self, symbol: str):
        """Initialize with symbol."""
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
    
    def get_comprehensive_earnings_data(self, lookback_years: int = 3) -> Dict[str, Any]:
        """
        Get all available earnings data for market inefficiency analysis.
        
        This method combines all the reliable yfinance earnings capabilities
        we discovered in our research.
        """
        print(f"📊 Collecting comprehensive earnings data for {self.symbol}")
        
        result = {
            'symbol': self.symbol,
            'collection_timestamp': datetime.now().isoformat(),
            'data_sources': [],
            'analysis_ready': False
        }
        
        try:
            # 1. Historical earnings dates with surprises (MOST RELIABLE)
            earnings_dates = self._get_earnings_dates_with_analysis()
            if earnings_dates is not None:
                result['earnings_dates'] = earnings_dates
                result['data_sources'].append('earnings_dates')
                print(f"   ✅ Earnings dates: {len(earnings_dates)} events")
            
            # 2. Upcoming earnings calendar
            calendar = self._get_upcoming_earnings()
            if calendar is not None:
                result['upcoming_earnings'] = calendar
                result['data_sources'].append('calendar')
                print(f"   ✅ Upcoming earnings: Available")
            
            # 3. Current analyst expectations
            analyst_data = self._get_analyst_data()
            if analyst_data is not None:
                result['analyst_data'] = analyst_data
                result['data_sources'].append('analyst_targets')
                print(f"   ✅ Analyst data: Available")
            
            # 4. Price data for event studies
            price_data = self._get_price_data_for_analysis(lookback_years)
            if price_data is not None:
                result['price_data'] = price_data
                result['data_sources'].append('price_data')
                print(f"   ✅ Price data: {len(price_data)} days")
            
            # 5. Generate market inefficiency features
            if 'earnings_dates' in result and 'price_data' in result:
                inefficiency_features = self._generate_inefficiency_features(
                    result['earnings_dates'], 
                    result['price_data']
                )
                result['inefficiency_features'] = inefficiency_features
                result['analysis_ready'] = True
                print(f"   ✅ Inefficiency features: {len(inefficiency_features)} events analyzed")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error collecting earnings data: {e}")
            result['error'] = str(e)
            return result
    
    def _get_earnings_dates_with_analysis(self) -> Optional[Dict[str, Any]]:
        """Get earnings dates with comprehensive analysis."""
        try:
            earnings_dates = self.ticker.earnings_dates
            
            if earnings_dates is None or earnings_dates.empty:
                return None
            
            # Clean and analyze the data
            df = earnings_dates.copy()
            
            # Add timing analysis
            df['day_of_week'] = df.index.day_name()
            df['is_tuesday_friday'] = df['day_of_week'].isin(['Tuesday', 'Friday'])
            df['quarter'] = df.index.to_series().dt.quarter
            df['month'] = df.index.to_series().dt.month
            df['is_earnings_season'] = df['month'].isin([1, 4, 7, 10])
            
            # Surprise analysis
            if 'Surprise(%)' in df.columns:
                df['large_surprise'] = abs(df['Surprise(%)']) > 5
                df['positive_surprise'] = df['Surprise(%)'] > 0
                df['surprise_magnitude'] = abs(df['Surprise(%)'])
            
            # Calculate statistics
            stats = self._calculate_earnings_statistics(df)
            
            return {
                'data': df,
                'statistics': stats,
                'date_range': {
                    'start': df.index.min(),
                    'end': df.index.max(),
                    'years_of_data': (df.index.max() - df.index.min()).days / 365.25
                }
            }
            
        except Exception as e:
            print(f"   ❌ Error getting earnings dates: {e}")
            return None
    
    def _get_upcoming_earnings(self) -> Optional[Dict[str, Any]]:
        """Get upcoming earnings with timing analysis."""
        try:
            calendar = self.ticker.calendar
            
            if calendar is None:
                return None
            
            result = {}
            
            if isinstance(calendar, dict):
                if 'Earnings Date' in calendar:
                    next_earnings = calendar['Earnings Date']
                    if isinstance(next_earnings, list) and next_earnings:
                        next_date = next_earnings[0]
                        current_date = datetime.now().date()
                        days_to_earnings = (next_date - current_date).days
                        
                        result.update({
                            'next_earnings_date': next_date,
                            'days_to_earnings': days_to_earnings,
                            'day_of_week': next_date.strftime('%A'),
                            'is_tuesday_friday': next_date.strftime('%A') in ['Tuesday', 'Friday'],
                            'earnings_proximity_zone': self._classify_earnings_proximity(days_to_earnings)
                        })
                
                # Add estimate data
                if 'Earnings Average' in calendar:
                    result['eps_estimate'] = calendar['Earnings Average']
                if 'Earnings High' in calendar:
                    result['eps_estimate_high'] = calendar['Earnings High']
                if 'Earnings Low' in calendar:
                    result['eps_estimate_low'] = calendar['Earnings Low']
                if 'Revenue Average' in calendar:
                    result['revenue_estimate'] = calendar['Revenue Average']
            
            return result if result else None
            
        except Exception as e:
            print(f"   ❌ Error getting calendar: {e}")
            return None
    
    def _get_analyst_data(self) -> Optional[Dict[str, Any]]:
        """Get current analyst data."""
        try:
            targets = self.ticker.analyst_price_targets
            
            if targets is None:
                return None
            
            result = {}
            
            if isinstance(targets, dict):
                result.update(targets)
            elif hasattr(targets, 'to_dict'):
                result = targets.to_dict()
            
            return result if result else None
            
        except Exception as e:
            print(f"   ❌ Error getting analyst data: {e}")
            return None
    
    def _get_price_data_for_analysis(self, lookback_years: int) -> Optional[pd.DataFrame]:
        """Get price data with technical indicators for earnings analysis."""
        try:
            # Get price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_years * 365)
            
            price_data = yf.download(self.symbol, start=start_date, end=end_date, progress=False)
            
            if price_data.empty:
                return None
            
            # Add technical indicators for earnings analysis
            price_data['daily_return'] = price_data['Close'].pct_change()
            price_data['abs_return'] = abs(price_data['daily_return'])
            price_data['log_return'] = np.log(price_data['Close'] / price_data['Close'].shift(1))
            
            # Volatility measures
            price_data['volatility_5d'] = price_data['daily_return'].rolling(5).std() * np.sqrt(252)
            price_data['volatility_20d'] = price_data['daily_return'].rolling(20).std() * np.sqrt(252)
            price_data['volatility_60d'] = price_data['daily_return'].rolling(60).std() * np.sqrt(252)
            
            # Moving averages
            price_data['sma_20'] = price_data['Close'].rolling(20).mean()
            price_data['sma_50'] = price_data['Close'].rolling(50).mean()
            price_data['ema_12'] = price_data['Close'].ewm(span=12).mean()
            
            # Distance from moving averages (mean reversion indicators)
            price_data['distance_from_sma20'] = (price_data['Close'] - price_data['sma_20']) / price_data['sma_20']
            price_data['distance_from_sma50'] = (price_data['Close'] - price_data['sma_50']) / price_data['sma_50']
            
            # Volume analysis (if available)
            if 'Volume' in price_data.columns:
                price_data['volume_sma_20'] = price_data['Volume'].rolling(20).mean()
                price_data['volume_ratio'] = price_data['Volume'] / price_data['volume_sma_20']
            
            # Forward returns for analysis (don't use these for live trading!)
            for days in [1, 3, 5, 10, 20]:
                price_data[f'forward_return_{days}d'] = price_data['Close'].pct_change(periods=-days)
            
            return price_data
            
        except Exception as e:
            print(f"   ❌ Error getting price data: {e}")
            return None
    
    def _generate_inefficiency_features(self, earnings_data: Dict, price_data: pd.DataFrame) -> List[Dict]:
        """Generate features for market inefficiency analysis around earnings."""
        features = []
        
        earnings_df = earnings_data['data']
        
        for earnings_date, earnings_row in earnings_df.iterrows():
            try:
                # Find closest trading day
                earnings_date_only = earnings_date.date()
                price_dates = price_data.index.date
                
                closest_idx = None
                min_diff = float('inf')
                
                for i, price_date in enumerate(price_dates):
                    diff = abs((price_date - earnings_date_only).days)
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = i
                
                if closest_idx is None or min_diff > 5:  # Skip if >5 days away
                    continue
                
                # Extract earnings event features
                event_features = {
                    'earnings_date': earnings_date,
                    'symbol': self.symbol,
                    'surprise_pct': earnings_row.get('Surprise(%)', np.nan),
                    'eps_estimate': earnings_row.get('EPS Estimate', np.nan),
                    'reported_eps': earnings_row.get('Reported EPS', np.nan),
                    'day_of_week': earnings_row['day_of_week'],
                    'is_tuesday_friday': earnings_row['is_tuesday_friday'],
                    'is_earnings_season': earnings_row['is_earnings_season'],
                    'quarter': earnings_row['quarter']
                }
                
                # Add surprise classification
                surprise = earnings_row.get('Surprise(%)', np.nan)
                if not pd.isna(surprise):
                    event_features.update({
                        'large_surprise': abs(surprise) > 5,
                        'huge_surprise': abs(surprise) > 10,
                        'positive_surprise': surprise > 0,
                        'surprise_magnitude': abs(surprise)
                    })
                
                # Pre-earnings features (10 days before)
                pre_start = max(0, closest_idx - 10)
                pre_end = closest_idx
                if pre_end > pre_start:
                    pre_data = price_data.iloc[pre_start:pre_end]
                    
                    event_features.update({
                        'pre_earnings_return_5d': pre_data['daily_return'].tail(5).sum(),
                        'pre_earnings_return_10d': pre_data['daily_return'].sum(),
                        'pre_earnings_volatility': pre_data['abs_return'].mean(),
                        'pre_earnings_volume_spike': pre_data.get('volume_ratio', pd.Series([1])).mean()
                    })
                
                # Post-earnings features (20 days after)
                post_start = closest_idx + 1
                post_end = min(len(price_data), post_start + 20)
                if post_end > post_start:
                    post_data = price_data.iloc[post_start:post_end]
                    
                    # Different drift periods
                    for days in [1, 3, 5, 10, 20]:
                        if len(post_data) >= days:
                            drift_return = post_data['daily_return'].iloc[:days].sum()
                            event_features[f'post_earnings_drift_{days}d'] = drift_return
                    
                    # Volatility analysis
                    event_features.update({
                        'post_earnings_volatility': post_data['abs_return'].mean(),
                        'volatility_ratio_post_pre': (post_data['abs_return'].mean() / 
                                                    pre_data['abs_return'].mean() 
                                                    if 'pre_data' in locals() and pre_data['abs_return'].mean() > 0 
                                                    else np.nan)
                    })
                
                # Earnings day features
                earnings_day_data = price_data.iloc[closest_idx]
                event_features.update({
                    'earnings_day_return': earnings_day_data['daily_return'],
                    'earnings_day_volatility': earnings_day_data['abs_return'],
                    'earnings_day_volume_spike': earnings_day_data.get('volume_ratio', np.nan)
                })
                
                features.append(event_features)
                
            except Exception as e:
                print(f"   ⚠️ Error processing earnings event {earnings_date}: {e}")
                continue
        
        return features
    
    def _calculate_earnings_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive earnings statistics."""
        stats = {
            'total_events': len(df),
            'date_range_years': (df.index.max() - df.index.min()).days / 365.25
        }
        
        # Surprise statistics
        if 'Surprise(%)' in df.columns:
            surprises = df['Surprise(%)'].dropna()
            if not surprises.empty:
                stats.update({
                    'mean_surprise': surprises.mean(),
                    'median_surprise': surprises.median(),
                    'std_surprise': surprises.std(),
                    'positive_surprise_rate': (surprises > 0).mean(),
                    'large_surprise_rate': (abs(surprises) > 5).mean(),
                    'huge_surprise_rate': (abs(surprises) > 10).mean(),
                    'max_positive_surprise': surprises.max(),
                    'max_negative_surprise': surprises.min()
                })
        
        # Timing statistics
        day_distribution = df['day_of_week'].value_counts()
        stats['day_of_week_distribution'] = day_distribution.to_dict()
        stats['tuesday_friday_rate'] = df['is_tuesday_friday'].mean()
        
        # Quarterly distribution
        quarter_distribution = df['quarter'].value_counts()
        stats['quarterly_distribution'] = quarter_distribution.to_dict()
        
        return stats
    
    def _classify_earnings_proximity(self, days_to_earnings: int) -> str:
        """Classify proximity to earnings for analysis."""
        if days_to_earnings < 0:
            return 'post_earnings'
        elif days_to_earnings <= 2:
            return 'immediate_pre_earnings'
        elif days_to_earnings <= 7:
            return 'near_term_pre_earnings'
        elif days_to_earnings <= 21:
            return 'mid_term_pre_earnings'
        else:
            return 'distant_from_earnings'
    
    def analyze_post_earnings_drift(self) -> Dict[str, Any]:
        """Specific analysis of post-earnings drift patterns."""
        print(f"📈 Analyzing post-earnings drift for {self.symbol}")
        
        # Get comprehensive data
        data = self.get_comprehensive_earnings_data()
        
        if not data.get('analysis_ready', False):
            return {'error': 'Insufficient data for drift analysis'}
        
        features = data['inefficiency_features']
        
        # Analyze drift by surprise magnitude
        drift_analysis = {
            'all_events': [],
            'positive_surprises': [],
            'negative_surprises': [],
            'large_surprises': [],
            'tuesday_friday_events': [],
            'other_day_events': []
        }
        
        for event in features:
            surprise = event.get('surprise_pct', np.nan)
            
            # All events
            event_summary = {
                'earnings_date': event['earnings_date'],
                'surprise': surprise,
                'drift_3d': event.get('post_earnings_drift_3d', np.nan),
                'drift_10d': event.get('post_earnings_drift_10d', np.nan),
                'drift_20d': event.get('post_earnings_drift_20d', np.nan),
                'day_of_week': event['day_of_week'],
                'is_tuesday_friday': event['is_tuesday_friday']
            }
            
            drift_analysis['all_events'].append(event_summary)
            
            # Categorize by surprise
            if not pd.isna(surprise):
                if surprise > 0:
                    drift_analysis['positive_surprises'].append(event_summary)
                else:
                    drift_analysis['negative_surprises'].append(event_summary)
                
                if abs(surprise) > 5:
                    drift_analysis['large_surprises'].append(event_summary)
            
            # Categorize by day
            if event['is_tuesday_friday']:
                drift_analysis['tuesday_friday_events'].append(event_summary)
            else:
                drift_analysis['other_day_events'].append(event_summary)
        
        # Calculate summary statistics
        summary = {}
        for category, events in drift_analysis.items():
            if events:
                df = pd.DataFrame(events)
                
                summary[category] = {
                    'count': len(df),
                    'mean_drift_3d': df['drift_3d'].mean() if 'drift_3d' in df.columns else np.nan,
                    'mean_drift_10d': df['drift_10d'].mean() if 'drift_10d' in df.columns else np.nan,
                    'mean_drift_20d': df['drift_20d'].mean() if 'drift_20d' in df.columns else np.nan,
                    'positive_drift_rate_10d': (df['drift_10d'] > 0).mean() if 'drift_10d' in df.columns else np.nan
                }
        
        return {
            'symbol': self.symbol,
            'analysis_timestamp': datetime.now().isoformat(),
            'drift_analysis': drift_analysis,
            'summary_statistics': summary
        }

# Example usage function
def demonstrate_enhanced_earnings_analysis():
    """Demonstrate the enhanced earnings analysis."""
    print("🎯 ENHANCED EARNINGS ANALYSIS DEMONSTRATION")
    print("=" * 50)
    
    # Test with multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        print(f"\n🔍 Analyzing {symbol}...")
        
        analyzer = EnhancedEarningsAnalyzer(symbol)
        
        # Get comprehensive data
        data = analyzer.get_comprehensive_earnings_data()
        
        if data.get('analysis_ready', False):
            print(f"   ✅ Analysis ready with {len(data['data_sources'])} data sources")
            
            # Show key statistics
            if 'inefficiency_features' in data:
                features = data['inefficiency_features']
                print(f"   📊 Analyzed {len(features)} earnings events")
                
                # Quick drift analysis
                drift_10d = [f.get('post_earnings_drift_10d', np.nan) for f in features 
                           if not pd.isna(f.get('post_earnings_drift_10d', np.nan))]
                
                if drift_10d:
                    mean_drift = np.mean(drift_10d)
                    positive_rate = sum(1 for d in drift_10d if d > 0) / len(drift_10d)
                    print(f"   📈 Mean 10d drift: {mean_drift:.2%}")
                    print(f"   📈 Positive drift rate: {positive_rate:.1%}")
            
            # Analyze post-earnings drift patterns
            drift_analysis = analyzer.analyze_post_earnings_drift()
            if 'summary_statistics' in drift_analysis:
                stats = drift_analysis['summary_statistics']
                if 'all_events' in stats:
                    all_stats = stats['all_events']
                    print(f"   🎯 Overall 10d drift: {all_stats.get('mean_drift_10d', 0):.2%}")
                    
                    # Show Tuesday/Friday effect if available
                    if 'tuesday_friday_events' in stats and 'other_day_events' in stats:
                        tf_drift = stats['tuesday_friday_events'].get('mean_drift_10d', 0)
                        other_drift = stats['other_day_events'].get('mean_drift_10d', 0)
                        print(f"   📅 Tuesday/Friday drift: {tf_drift:.2%}")
                        print(f"   📅 Other days drift: {other_drift:.2%}")
        else:
            print(f"   ❌ Insufficient data for analysis")

if __name__ == "__main__":
    demonstrate_enhanced_earnings_analysis()