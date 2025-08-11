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
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

def analyze_earnings_capabilities():
    """Quick analysis of yfinance earnings capabilities."""
    print("🎯 YFINANCE EARNINGS ANALYSIS CAPABILITIES")
    print("=" * 50)
    
    symbol = "AAPL"
    ticker = yf.Ticker(symbol)
    
    print(f"\n📊 Testing {symbol} earnings data:")
    print("-" * 30)
    
    # Test earnings_dates - most reliable method
    try:
        earnings_dates = ticker.earnings_dates
        if earnings_dates is not None and not earnings_dates.empty:
            print(f"✅ earnings_dates: {len(earnings_dates)} records")
            print(f"   📅 Date range: {earnings_dates.index.min()} to {earnings_dates.index.max()}")
            print(f"   📊 Columns: {list(earnings_dates.columns)}")
            
            # Show sample data
            print(f"\n📋 Sample earnings data:")
            print(earnings_dates.head())
            
            # Analyze surprises
            if 'Surprise(%)' in earnings_dates.columns:
                surprises = earnings_dates['Surprise(%)'].dropna()
                if not surprises.empty:
                    print(f"\n📈 Surprise Analysis:")
                    print(f"   Mean surprise: {surprises.mean():.2f}%")
                    print(f"   Std surprise: {surprises.std():.2f}%")
                    print(f"   Positive surprises: {(surprises > 0).sum()}/{len(surprises)} ({(surprises > 0).mean():.1%})")
                    print(f"   Large surprises (>5%): {(abs(surprises) > 5).sum()}/{len(surprises)} ({(abs(surprises) > 5).mean():.1%})")
        
    except Exception as e:
        print(f"❌ earnings_dates error: {e}")
    
    # Test calendar - upcoming earnings
    try:
        calendar = ticker.calendar
        if calendar is not None:
            print(f"\n✅ calendar: Available")
            if isinstance(calendar, dict):
                if 'Earnings Date' in calendar:
                    next_earnings = calendar['Earnings Date']
                    print(f"   📅 Next earnings: {next_earnings}")
                if 'Earnings Average' in calendar:
                    print(f"   💰 EPS estimate: {calendar['Earnings Average']}")
    except Exception as e:
        print(f"❌ calendar error: {e}")
    
    # Test price data integration
    try:
        print(f"\n📈 Price data integration test:")
        price_data = yf.download(symbol, period="1y", progress=False)
        if not price_data.empty:
            print(f"   ✅ Price data: {len(price_data)} days")
            
            # Calculate returns and volatility
            price_data['daily_return'] = price_data['Close'].pct_change()
            price_data['volatility'] = price_data['daily_return'].rolling(20).std() * np.sqrt(252)
            
            print(f"   📊 Mean daily return: {price_data['daily_return'].mean():.3%}")
            print(f"   📊 Volatility (20d): {price_data['volatility'].iloc[-1]:.1%}")
            
    except Exception as e:
        print(f"❌ price data error: {e}")

def demonstrate_earnings_event_analysis():
    """Demonstrate complete earnings event analysis workflow."""
    print(f"\n\n🎯 EARNINGS EVENT ANALYSIS WORKFLOW")
    print("=" * 45)
    
    symbol = "MSFT"  # Good example with reliable data
    ticker = yf.Ticker(symbol)
    
    print(f"🔄 Analyzing {symbol} earnings events:")
    print("-" * 35)
    
    try:
        # Step 1: Get earnings dates
        earnings_dates = ticker.earnings_dates
        if earnings_dates is None or earnings_dates.empty:
            print("❌ No earnings data available")
            return
        
        print(f"1️⃣ Found {len(earnings_dates)} earnings events")
        
        # Step 2: Get price data
        price_data = yf.download(symbol, period="3y", progress=False)
        if price_data.empty:
            print("❌ No price data available")
            return
        
        price_data['daily_return'] = price_data['Close'].pct_change()
        print(f"2️⃣ Downloaded {len(price_data)} days of price data")
        
        # Step 3: Analyze post-earnings drift
        print(f"3️⃣ Analyzing post-earnings patterns...")
        
        post_earnings_returns = []
        for earnings_date, row in earnings_dates.iterrows():
            surprise = row.get('Surprise(%)', np.nan)
            if pd.isna(surprise):
                continue
            
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
            
            if closest_idx is None or min_diff > 3:
                continue
            
            # Get post-earnings returns (next 10 days)
            start_idx = closest_idx + 1
            end_idx = min(len(price_data), start_idx + 10)
            
            if end_idx > start_idx:
                post_returns = price_data['daily_return'].iloc[start_idx:end_idx]
                cumulative_return = post_returns.sum()
                
                post_earnings_returns.append({
                    'earnings_date': earnings_date,
                    'surprise': surprise,
                    'post_10d_return': cumulative_return,
                    'large_surprise': abs(surprise) > 5
                })
        
        if post_earnings_returns:
            df = pd.DataFrame(post_earnings_returns)
            
            print(f"   📊 Analyzed {len(df)} earnings events")
            print(f"   📈 Mean 10-day post-earnings return: {df['post_10d_return'].mean():.2%}")
            print(f"   🎯 Positive return rate: {(df['post_10d_return'] > 0).mean():.1%}")
            
            # Analyze by surprise magnitude
            large_surprises = df[df['large_surprise']]
            if len(large_surprises) > 0:
                print(f"   ⚡ Large surprises (>5%): {len(large_surprises)} events")
                print(f"   ⚡ Large surprise 10d return: {large_surprises['post_10d_return'].mean():.2%}")
            
            # Correlation between surprise and returns
            correlation = df['surprise'].corr(df['post_10d_return'])
            print(f"   🔗 Surprise-return correlation: {correlation:.3f}")
        
        # Step 4: Day-of-week analysis
        print(f"4️⃣ Day-of-week analysis...")
        earnings_dates['day_of_week'] = earnings_dates.index.day_name()
        day_distribution = earnings_dates['day_of_week'].value_counts()
        
        print("   📅 Announcement day distribution:")
        for day, count in day_distribution.items():
            percentage = count / len(earnings_dates)
            print(f"      {day}: {count} announcements ({percentage:.1%})")
        
        # Tuesday/Friday pattern analysis
        tuesday_friday = earnings_dates[earnings_dates['day_of_week'].isin(['Tuesday', 'Friday'])]
        tf_rate = len(tuesday_friday) / len(earnings_dates)
        print(f"   🎯 Tuesday/Friday rate: {tf_rate:.1%}")
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")

def demonstrate_alternative_data_sources():
    """Demonstrate alternative/complementary data sources for earnings analysis."""
    print(f"\n\n💡 ALTERNATIVE DATA SOURCES FOR EARNINGS")
    print("=" * 45)
    
    print("""
🎯 YFINANCE LIMITATIONS & ALTERNATIVES:

❌ YFINANCE LIMITATIONS:
   • Historical earnings dates: Only ~2-3 years
   • No intraday earnings timing (pre/post market)
   • Limited analyst revision history
   • No earnings guidance data
   • No conference call sentiment

✅ COMPLEMENTARY DATA SOURCES:
   1. SEC EDGAR API (free):
      • 10-Q/10-K filings with exact timestamps
      • Historical earnings back to 1990s+
      • Management guidance in filings

   2. Alpha Vantage (free tier):
      • earnings() endpoint with historical data
      • Quarterly earnings going back 20+ years
      • Annual earnings with more history

   3. Financial Modeling Prep (free tier):
      • Historical earnings calendar
      • Earnings surprises with more history
      • Conference call transcripts

   4. Yahoo Finance RSS/News:
      • Real-time earnings announcements
      • Conference call times
      • Management guidance updates

🚀 ENHANCED ANALYSIS STRATEGY:
   1. Use yfinance for: Recent earnings (2022-2025), surprises, OHLCV data
   2. Use Alpha Vantage for: Extended historical earnings (2000+)
   3. Use SEC filings for: Exact filing times, guidance analysis
   4. Combine all sources for: Complete picture of earnings timing inefficiencies

📊 CODE EXAMPLE - Alpha Vantage Integration:
""")
    
    print('''
# Alpha Vantage earnings example (requires API key)
import requests

def get_extended_earnings_history(symbol, api_key):
    """Get 20+ years of earnings data from Alpha Vantage."""
    url = f"https://www.alphavantage.co/query"
    params = {
        'function': 'EARNINGS',
        'symbol': symbol,
        'apikey': api_key
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'quarterlyEarnings' in data:
        quarterly_earnings = pd.DataFrame(data['quarterlyEarnings'])
        quarterly_earnings['fiscalDateEnding'] = pd.to_datetime(quarterly_earnings['fiscalDateEnding'])
        quarterly_earnings['reportedEPS'] = pd.to_numeric(quarterly_earnings['reportedEPS'], errors='coerce')
        quarterly_earnings['estimatedEPS'] = pd.to_numeric(quarterly_earnings['estimatedEPS'], errors='coerce')
        quarterly_earnings['surprise'] = quarterly_earnings['reportedEPS'] - quarterly_earnings['estimatedEPS']
        quarterly_earnings['surprise_pct'] = (quarterly_earnings['surprise'] / quarterly_earnings['estimatedEPS'].abs()) * 100
        
        return quarterly_earnings
    
    return pd.DataFrame()

# Usage example:
# earnings_data = get_extended_earnings_history('AAPL', 'your_api_key')
# print(f"Historical earnings data: {len(earnings_data)} quarters")
# print(f"Date range: {earnings_data['fiscalDateEnding'].min()} to {earnings_data['fiscalDateEnding'].max()}")
''')

def generate_yfinance_best_practices():
    """Generate best practices summary for using yfinance for earnings analysis."""
    print(f"\n\n🚀 YFINANCE EARNINGS ANALYSIS BEST PRACTICES")
    print("=" * 55)
    
    best_practices = """
✅ OPTIMAL YFINANCE USAGE FOR EARNINGS:

1. 📅 HISTORICAL EARNINGS DATES (ticker.earnings_dates):
   • Reliable 2-3 years of data with exact announcement dates
   • Includes EPS estimates, actual EPS, and surprise percentages
   • Perfect for recent post-earnings drift analysis
   • Use for: Tuesday/Friday timing analysis, recent surprise patterns

2. 📊 UPCOMING EARNINGS (ticker.calendar):
   • Next earnings date and EPS estimates
   • Revenue estimates and ranges
   • Use for: Days-to-earnings proximity features

3. 💲 PRICE DATA INTEGRATION (yf.download):
   • Daily OHLCV data back 20+ years
   • Combine with earnings dates for event studies
   • Calculate: Pre/post earnings returns, volatility changes

4. 🎯 ANALYST DATA (ticker.analyst_price_targets):
   • Current analyst consensus
   • Price targets for expectation analysis
   • Limited historical depth

📈 PROVEN INEFFICIENCY PATTERNS TO ANALYZE:

1. POST-EARNINGS DRIFT:
   • 60-day post-earnings returns by surprise magnitude
   • Stronger drift for smaller cap stocks
   • Tuesday vs Friday announcement differences

2. PRE-EARNINGS RUN-UP:
   • 10-20 day pre-earnings price movements
   • Correlation with subsequent surprise
   • Volume patterns in pre-earnings period

3. TIMING INEFFICIENCIES:
   • Tuesday announcements vs Friday (weekend effect)
   • After-hours vs pre-market announcements
   • Earnings season clustering effects

4. SURPRISE MAGNITUDE EFFECTS:
   • >10% surprises often have extended drift
   • Negative surprises may have faster reversals
   • Guidance updates vs earnings beats

💡 FEATURE ENGINEERING IDEAS:

earnings_features = {
    'days_to_next_earnings': (next_earnings_date - current_date).days,
    'earnings_surprise_history': last_4_quarters_surprises.mean(),
    'surprise_consistency': last_4_quarters_surprises.std(),
    'beat_rate': (last_8_quarters_surprises > 0).mean(),
    'announcement_day_pattern': earnings_date.strftime('%A'),
    'earnings_season_indicator': current_month in [1, 4, 7, 10],
    'post_earnings_drift_20d': returns_20d_post_last_earnings,
    'pre_earnings_runup_10d': returns_10d_pre_earnings,
    'earnings_volatility_ratio': earnings_day_vol / normal_vol,
    'analyst_revision_trend': recent_estimate_changes.mean()
}

🎯 COMBINING WITH OHLCV FOR STRATEGY:

def create_earnings_features(symbol, lookback_days=252):
    ticker = yf.Ticker(symbol)
    
    # Get earnings and price data
    earnings_dates = ticker.earnings_dates
    price_data = yf.download(symbol, period="3y", progress=False)
    
    # Calculate technical indicators
    price_data['returns'] = price_data['Close'].pct_change()
    price_data['volatility_20d'] = price_data['returns'].rolling(20).std()
    price_data['sma_20'] = price_data['Close'].rolling(20).mean()
    
    features = []
    for date, price_row in price_data.iterrows():
        # Find next earnings
        future_earnings = earnings_dates[earnings_dates.index > date]
        if not future_earnings.empty:
            next_earnings = future_earnings.index[0]
            days_to_earnings = (next_earnings.date() - date.date()).days
            
            # Find last earnings
            past_earnings = earnings_dates[earnings_dates.index < date]
            if not past_earnings.empty:
                last_earnings = past_earnings.index[-1]
                last_surprise = past_earnings.loc[last_earnings, 'Surprise(%)']
                days_since_earnings = (date.date() - last_earnings.date()).days
                
                features.append({
                    'date': date,
                    'close': price_row['Close'],
                    'days_to_earnings': days_to_earnings,
                    'days_since_earnings': days_since_earnings,
                    'last_surprise': last_surprise,
                    'volatility_20d': price_row['volatility_20d'],
                    'distance_from_sma20': (price_row['Close'] - price_row['sma_20']) / price_row['sma_20'],
                    'earnings_proximity_zone': 'near' if days_to_earnings <= 5 else 'far'
                })
    
    return pd.DataFrame(features)
"""
    
    print(best_practices)

def main():
    """Run comprehensive yfinance earnings research."""
    print("🎯 COMPREHENSIVE YFINANCE EARNINGS RESEARCH")
    print("=" * 55)
    print("Researching yfinance capabilities for finding market inefficiencies around earnings events...")
    
    # Run all analyses
    analyze_earnings_capabilities()
    demonstrate_earnings_event_analysis() 
    demonstrate_alternative_data_sources()
    generate_yfinance_best_practices()
    
    # Final summary
    print(f"\n\n🎯 FINAL SUMMARY: YFINANCE FOR EARNINGS ANALYSIS")
    print("=" * 55)
    
    summary = """
✅ WHAT YFINANCE CAN DO FOR EARNINGS ANALYSIS:
   • 2-3 years of historical earnings dates with surprises ✓
   • Exact announcement dates and times ✓ 
   • EPS estimates vs actuals for surprise calculation ✓
   • Integration with daily OHLCV price data ✓
   • Upcoming earnings dates and estimates ✓
   • Current analyst price targets ✓

⚠️ YFINANCE LIMITATIONS:
   • Limited historical depth (~2-3 years) ❌
   • No intraday announcement timing ❌
   • No historical analyst revisions ❌
   • No earnings guidance tracking ❌
   • ticker.quarterly_earnings often empty ❌

🚀 RECOMMENDED APPROACH:
   1. Use yfinance as PRIMARY source for recent data (2022-2025)
   2. Supplement with Alpha Vantage for extended history
   3. Focus on these proven inefficiencies:
      • Post-earnings drift (10-60 days)
      • Tuesday vs Friday announcement patterns  
      • Earnings surprise magnitude effects
      • Pre-earnings run-up patterns
      • Volatility clustering around earnings

💡 BEST FEATURES FOR ML MODELS:
   • days_to_next_earnings (proximity effect)
   • historical_surprise_rate (consistency)
   • announcement_day_of_week (timing effect)
   • post_earnings_drift_20d (momentum)
   • earnings_volatility_spike (uncertainty)

This gives you a solid foundation for finding earnings-related market inefficiencies!
"""
    
    print(summary)

if __name__ == "__main__":
    main()