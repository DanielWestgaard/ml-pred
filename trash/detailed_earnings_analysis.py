#!/usr/bin/env python3
"""
Detailed analysis of yfinance earnings data structure and capabilities.

This script provides in-depth examples of what data is actually available
and how to use it for finding market inefficiencies around earnings events.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def analyze_earnings_dates_detailed():
    """Detailed analysis of ticker.earnings_dates capability."""
    print("🎯 DETAILED ANALYSIS: ticker.earnings_dates")
    print("=" * 50)
    
    # Test with different symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol} earnings dates:")
        print("-" * 30)
        
        ticker = yf.Ticker(symbol)
        try:
            earnings_dates = ticker.earnings_dates
            
            if earnings_dates is not None and not earnings_dates.empty:
                print(f"📅 Data shape: {earnings_dates.shape}")
                print(f"🗓️ Date range: {earnings_dates.index.min()} to {earnings_dates.index.max()}")
                print(f"📊 Columns: {list(earnings_dates.columns)}")
                
                # Show sample data
                print(f"\n🔍 Sample earnings data:")
                print(earnings_dates.head(8))
                
                # Analyze surprise data
                if 'Surprise(%)' in earnings_dates.columns:
                    surprises = earnings_dates['Surprise(%)'].dropna()
                    if not surprises.empty:
                        print(f"\n📈 Surprise Statistics:")
                        print(f"   Mean surprise: {surprises.mean():.2f}%")
                        print(f"   Std surprise: {surprises.std():.2f}%")
                        print(f"   Min surprise: {surprises.min():.2f}%")
                        print(f"   Max surprise: {surprises.max():.2f}%")
                        print(f"   Positive surprises: {(surprises > 0).sum()}/{len(surprises)} ({(surprises > 0).mean():.1%})")
                
                # Analyze timing patterns
                earnings_dates['day_of_week'] = earnings_dates.index.day_name()
                day_distribution = earnings_dates['day_of_week'].value_counts()
                print(f"\n📅 Earnings Day Distribution:")
                for day, count in day_distribution.items():
                    print(f"   {day}: {count} announcements ({count/len(earnings_dates):.1%})")
                
                # Check for quarterly patterns
                earnings_dates['quarter'] = earnings_dates.index.to_series().dt.quarter
                quarter_dist = earnings_dates['quarter'].value_counts().sort_index()
                print(f"\n🗓️ Quarterly Distribution:")
                for quarter, count in quarter_dist.items():
                    print(f"   Q{quarter}: {count} announcements ({count/len(earnings_dates):.1%})")
                
            else:
                print("❌ No earnings dates data available")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def analyze_calendar_data():
    """Analyze upcoming earnings calendar data."""
    print("\n\n🎯 DETAILED ANALYSIS: ticker.calendar")
    print("=" * 50)
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        print(f"\n📅 Analyzing {symbol} earnings calendar:")
        print("-" * 35)
        
        ticker = yf.Ticker(symbol)
        try:
            calendar = ticker.calendar
            
            if calendar is not None:
                print(f"📊 Calendar data type: {type(calendar)}")
                
                if hasattr(calendar, 'columns'):
                    print(f"📊 Shape: {calendar.shape}")
                    print(f"🗂️ Columns: {list(calendar.columns)}")
                    print(f"\n📋 Calendar data:")
                    print(calendar)
                elif isinstance(calendar, dict):
                    print(f"📊 Calendar dict keys: {list(calendar.keys())}")
                    for key, value in calendar.items():
                        print(f"   {key}: {value}")
                else:
                    print(f"📊 Calendar data: {calendar}")
            else:
                print("❌ No calendar data available")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def test_quarterly_earnings_alternative():
    """Test alternative methods to get quarterly earnings."""
    print("\n\n🎯 TESTING ALTERNATIVE QUARTERLY EARNINGS METHODS")
    print("=" * 55)
    
    symbol = "AAPL"
    ticker = yf.Ticker(symbol)
    
    print(f"🧪 Testing alternative methods for {symbol}:")
    print("-" * 40)
    
    # Method 1: Try quarterly_earnings directly
    try:
        quarterly_earnings = ticker.quarterly_earnings
        print(f"✅ ticker.quarterly_earnings: Available")
        if quarterly_earnings is not None and hasattr(quarterly_earnings, 'shape'):
            print(f"   📊 Shape: {quarterly_earnings.shape}")
            if hasattr(quarterly_earnings, 'columns'):
                print(f"   📋 Columns: {list(quarterly_earnings.columns)}")
            if not quarterly_earnings.empty:
                print(f"   📈 Sample data:")
                print(quarterly_earnings.head())
    except Exception as e:
        print(f"❌ ticker.quarterly_earnings error: {e}")
    
    # Method 2: Try earnings property
    try:
        earnings = ticker.earnings
        print(f"✅ ticker.earnings: Available")
        if earnings is not None and hasattr(earnings, 'shape'):
            print(f"   📊 Shape: {earnings.shape}")
            if hasattr(earnings, 'columns'):
                print(f"   📋 Columns: {list(earnings.columns)}")
            if not earnings.empty:
                print(f"   📈 Sample data:")
                print(earnings.head())
    except Exception as e:
        print(f"❌ ticker.earnings error: {e}")
    
    # Method 3: Financial statements for earnings context
    try:
        financials = ticker.quarterly_financials
        print(f"✅ ticker.quarterly_financials: Available")
        if financials is not None and hasattr(financials, 'shape'):
            print(f"   📊 Shape: {financials.shape}")
            # Look for earnings-related rows
            earnings_rows = [row for row in financials.index if any(term in row.lower() for term in ['income', 'earnings', 'eps'])]
            if earnings_rows:
                print(f"   💰 Earnings-related metrics available: {earnings_rows[:5]}")
                print(f"   📈 Sample earnings data:")
                print(financials.loc[earnings_rows[:3]])
    except Exception as e:
        print(f"❌ ticker.quarterly_financials error: {e}")

def demonstrate_earnings_analysis_workflow():
    """Demonstrate a complete workflow for earnings analysis."""
    print("\n\n🎯 COMPLETE EARNINGS ANALYSIS WORKFLOW")
    print("=" * 45)
    
    symbol = "AAPL"
    ticker = yf.Ticker(symbol)
    
    print(f"🔄 Complete workflow for {symbol}:")
    print("-" * 30)
    
    try:
        # Step 1: Get historical earnings dates with surprises
        print("1️⃣ Getting historical earnings dates...")
        earnings_dates = ticker.earnings_dates
        
        if earnings_dates is not None and not earnings_dates.empty:
            print(f"   ✅ Found {len(earnings_dates)} historical earnings events")
            
            # Step 2: Calculate surprise statistics
            print("2️⃣ Analyzing earnings surprises...")
            if 'Surprise(%)' in earnings_dates.columns:
                surprises = earnings_dates['Surprise(%)'].dropna()
                large_surprises = surprises[abs(surprises) > 5]  # Surprises > 5%
                print(f"   📊 Large surprises (>5%): {len(large_surprises)}/{len(surprises)} ({len(large_surprises)/len(surprises):.1%})")
                
                # Find biggest surprises
                biggest_positive = surprises.max()
                biggest_negative = surprises.min()
                print(f"   📈 Biggest positive surprise: {biggest_positive:.1f}%")
                print(f"   📉 Biggest negative surprise: {biggest_negative:.1f}%")
        
        # Step 3: Get upcoming earnings
        print("3️⃣ Getting upcoming earnings...")
        calendar = ticker.calendar
        if calendar is not None:
            print(f"   📅 Upcoming earnings available")
            if hasattr(calendar, 'index') and not calendar.empty:
                next_earnings = calendar.index[0]
                days_to_earnings = (pd.to_datetime(next_earnings) - datetime.now()).days
                print(f"   🗓️ Next earnings: {next_earnings} ({days_to_earnings} days away)")
        
        # Step 4: Get price data for context
        print("4️⃣ Getting price context...")
        # Get 1 year of price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        price_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if not price_data.empty:
            current_price = price_data['Close'].iloc[-1]
            yearly_high = price_data['High'].max()
            yearly_low = price_data['Low'].min()
            
            print(f"   💲 Current price: ${current_price:.2f}")
            print(f"   📈 52-week range: ${yearly_low:.2f} - ${yearly_high:.2f}")
            
            # Calculate position relative to range
            position_in_range = (current_price - yearly_low) / (yearly_high - yearly_low)
            print(f"   📊 Position in 52-week range: {position_in_range:.1%}")
        
        # Step 5: Demonstrate earnings event analysis
        print("5️⃣ Earnings event timing analysis...")
        if not earnings_dates.empty:
            # Analyze day-of-week patterns
            earnings_dates['day_of_week'] = earnings_dates.index.day_name()
            
            # Tuesday/Friday hypothesis testing
            tuesday_friday = earnings_dates[earnings_dates['day_of_week'].isin(['Tuesday', 'Friday'])]
            tf_percentage = len(tuesday_friday) / len(earnings_dates)
            
            print(f"   📅 Tuesday/Friday announcements: {len(tuesday_friday)}/{len(earnings_dates)} ({tf_percentage:.1%})")
            
            # Surprise analysis by day
            if 'Surprise(%)' in earnings_dates.columns:
                tue_fri_surprises = tuesday_friday['Surprise(%)'].dropna()
                other_surprises = earnings_dates[~earnings_dates['day_of_week'].isin(['Tuesday', 'Friday'])]['Surprise(%)'].dropna()
                
                if not tue_fri_surprises.empty and not other_surprises.empty:
                    print(f"   📊 Tuesday/Friday avg surprise: {tue_fri_surprises.mean():.2f}%")
                    print(f"   📊 Other days avg surprise: {other_surprises.mean():.2f}%")
        
        print("\n✅ Workflow complete! This demonstrates how to:")
        print("   • Get historical earnings dates (2-3 years)")
        print("   • Calculate earnings surprises")
        print("   • Analyze timing patterns (day of week)")
        print("   • Get upcoming earnings dates")
        print("   • Combine with price data for context")
        
    except Exception as e:
        print(f"❌ Workflow error: {e}")

def test_extended_historical_data():
    """Test if we can get more historical data using different approaches."""
    print("\n\n🎯 TESTING EXTENDED HISTORICAL DATA ACCESS")
    print("=" * 45)
    
    symbol = "AAPL"
    ticker = yf.Ticker(symbol)
    
    # Test 1: Try to get earnings dates with extended period
    try:
        print("🧪 Testing extended earnings_dates access:")
        
        # Try different approaches to get more historical data
        earnings_dates = ticker.earnings_dates
        
        if earnings_dates is not None:
            print(f"   📅 Standard method gives: {len(earnings_dates)} records")
            print(f"   🗓️ Date range: {earnings_dates.index.min()} to {earnings_dates.index.max()}")
            
            # Check if there are any parameters or methods to extend this
            years_of_data = (earnings_dates.index.max() - earnings_dates.index.min()).days / 365.25
            print(f"   📊 Years of data: {years_of_data:.1f}")
            
            # Test if we can filter/extend
            earliest_date = earnings_dates.index.min()
            latest_date = earnings_dates.index.max()
            
            print(f"   🔍 Earliest earnings: {earliest_date}")
            print(f"   🔍 Latest earnings: {latest_date}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Compare with financial statements dates
    try:
        print("\n🧪 Comparing with financial statements timing:")
        
        quarterly_financials = ticker.quarterly_financials
        if quarterly_financials is not None and not quarterly_financials.empty:
            fin_dates = quarterly_financials.columns
            print(f"   📊 Quarterly financials dates: {len(fin_dates)} periods")
            if len(fin_dates) > 0:
                print(f"   🗓️ Financial date range: {fin_dates.min()} to {fin_dates.max()}")
                
                # Compare coverage
                fin_years = (fin_dates.max() - fin_dates.min()).days / 365.25
                print(f"   📅 Financial data years: {fin_years:.1f}")
    
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Run all detailed analyses."""
    analyze_earnings_dates_detailed()
    analyze_calendar_data()
    test_quarterly_earnings_alternative()
    demonstrate_earnings_analysis_workflow()
    test_extended_historical_data()
    
    print("\n" + "=" * 60)
    print("🎯 KEY FINDINGS FOR EARNINGS ANALYSIS")
    print("=" * 60)
    print("""
✅ AVAILABLE HISTORICAL DATA (2022-2025):
   • ticker.earnings_dates: 2-3 years of earnings announcement dates
   • Surprise data: Actual vs Expected EPS with percentage surprise
   • Day-of-week timing: Can analyze Tuesday/Friday patterns
   • Quarterly patterns: Can identify seasonal announcement timing

✅ REAL-TIME/UPCOMING DATA:
   • ticker.calendar: Next earnings announcement date
   • analyst_price_targets: Current analyst expectations
   • recommendations: Recent analyst sentiment changes

✅ FINANCIAL CONTEXT:
   • ticker.quarterly_financials: Quarterly earnings metrics
   • ticker.financials: Annual earnings context
   • Price data integration: Pre/post earnings price movements

⚠️ LIMITATIONS:
   • Historical depth limited to ~2-3 years for earnings_dates
   • ticker.quarterly_earnings often returns empty
   • Analyst revision history is limited
   • No access to intraday earnings announcement timing

🚀 BEST PRACTICES FOR MARKET INEFFICIENCIES:
   1. Use earnings_dates for historical surprise analysis
   2. Combine with OHLCV data for post-earnings drift
   3. Analyze Tuesday/Friday announcement patterns
   4. Monitor earnings proximity (days to next earnings)
   5. Track earnings season timing effects
   6. Use analyst_price_targets for expectation changes
""")

if __name__ == "__main__":
    main()