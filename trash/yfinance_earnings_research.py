#!/usr/bin/env python3
"""
Comprehensive research into yfinance's historical earnings capabilities.

This script tests various yfinance methods to understand what historical 
earnings data is actually available for finding market inefficiencies 
around earnings events.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
import json
from typing import Dict, Any, Optional, List

warnings.filterwarnings('ignore')

class YFinanceEarningsResearcher:
    """Research yfinance earnings capabilities systematically."""
    
    def __init__(self, symbols: List[str] = None):
        """Initialize with test symbols."""
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"]
        self.results = {}
    
    def research_all_capabilities(self) -> Dict[str, Any]:
        """Research all yfinance earnings capabilities."""
        print("=== YFINANCE HISTORICAL EARNINGS RESEARCH ===")
        print(f"Testing capabilities for symbols: {', '.join(self.symbols)}")
        print("=" * 60)
        
        for symbol in self.symbols:
            print(f"\n🔍 RESEARCHING {symbol}")
            print("-" * 40)
            
            ticker = yf.Ticker(symbol)
            symbol_results = {}
            
            # Test all available methods
            symbol_results['calendar'] = self._test_calendar(ticker, symbol)
            symbol_results['quarterly_earnings'] = self._test_quarterly_earnings(ticker, symbol)
            symbol_results['earnings_dates'] = self._test_earnings_dates(ticker, symbol)
            symbol_results['earnings_history'] = self._test_earnings_history(ticker, symbol)
            symbol_results['analyst_price_targets'] = self._test_analyst_targets(ticker, symbol)
            symbol_results['recommendations'] = self._test_recommendations(ticker, symbol)
            symbol_results['earnings_estimate'] = self._test_earnings_estimate(ticker, symbol)
            symbol_results['financials'] = self._test_financials(ticker, symbol)
            
            self.results[symbol] = symbol_results
        
        # Generate comprehensive summary
        self._generate_summary()
        return self.results
    
    def _test_calendar(self, ticker, symbol: str) -> Dict[str, Any]:
        """Test ticker.calendar for upcoming earnings."""
        print("📅 Testing: ticker.calendar")
        try:
            calendar = ticker.calendar
            result = {
                'available': calendar is not None,
                'data_type': str(type(calendar)),
                'sample_data': None,
                'columns': None,
                'date_range': None
            }
            
            if calendar is not None:
                if hasattr(calendar, 'columns'):
                    result['columns'] = list(calendar.columns)
                    result['sample_data'] = calendar.head().to_dict() if not calendar.empty else "Empty DataFrame"
                    
                    # Check for date columns
                    date_cols = [col for col in calendar.columns if 'date' in col.lower()]
                    if date_cols and not calendar.empty:
                        dates = pd.to_datetime(calendar[date_cols[0]], errors='coerce').dropna()
                        if not dates.empty:
                            result['date_range'] = f"{dates.min()} to {dates.max()}"
                
                elif isinstance(calendar, dict):
                    result['sample_data'] = calendar
                else:
                    result['sample_data'] = str(calendar)[:500]
            
            print(f"   ✅ Available: {result['available']}")
            if result['columns']:
                print(f"   📊 Columns: {result['columns']}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'available': False, 'error': str(e)}
    
    def _test_quarterly_earnings(self, ticker, symbol: str) -> Dict[str, Any]:
        """Test ticker.quarterly_earnings for historical data."""
        print("📈 Testing: ticker.quarterly_earnings")
        try:
            earnings = ticker.quarterly_earnings
            result = {
                'available': earnings is not None and (not hasattr(earnings, 'empty') or not earnings.empty),
                'data_type': str(type(earnings)),
                'shape': None,
                'columns': None,
                'date_range': None,
                'sample_data': None,
                'has_estimates': False,
                'has_actuals': False,
                'surprise_calculable': False
            }
            
            if earnings is not None and hasattr(earnings, 'columns'):
                result['shape'] = earnings.shape
                result['columns'] = list(earnings.columns)
                result['sample_data'] = earnings.head().to_dict() if not earnings.empty else "Empty DataFrame"
                
                # Check for key columns for surprise analysis
                result['has_estimates'] = any('estimate' in col.lower() for col in earnings.columns)
                result['has_actuals'] = any('actual' in col.lower() for col in earnings.columns)
                result['surprise_calculable'] = result['has_estimates'] and result['has_actuals']
                
                # Try to get date range from index
                if hasattr(earnings.index, 'min'):
                    try:
                        result['date_range'] = f"{earnings.index.min()} to {earnings.index.max()}"
                    except:
                        pass
            
            print(f"   ✅ Available: {result['available']}")
            if result['columns']:
                print(f"   📊 Columns: {result['columns']}")
                print(f"   📈 Shape: {result['shape']}")
                print(f"   🎯 Surprise calculable: {result['surprise_calculable']}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'available': False, 'error': str(e)}
    
    def _test_earnings_dates(self, ticker, symbol: str) -> Dict[str, Any]:
        """Test ticker.earnings_dates for historical announcement dates."""
        print("📅 Testing: ticker.earnings_dates")
        try:
            earnings_dates = ticker.earnings_dates
            result = {
                'available': earnings_dates is not None and (not hasattr(earnings_dates, 'empty') or not earnings_dates.empty),
                'data_type': str(type(earnings_dates)),
                'shape': None,
                'columns': None,
                'date_range': None,
                'sample_data': None,
                'historical_depth': None
            }
            
            if earnings_dates is not None and hasattr(earnings_dates, 'columns'):
                result['shape'] = earnings_dates.shape
                result['columns'] = list(earnings_dates.columns)
                result['sample_data'] = earnings_dates.head().to_dict() if not earnings_dates.empty else "Empty DataFrame"
                
                # Check date range and historical depth
                if hasattr(earnings_dates.index, 'min'):
                    try:
                        min_date = earnings_dates.index.min()
                        max_date = earnings_dates.index.max()
                        result['date_range'] = f"{min_date} to {max_date}"
                        
                        # Calculate years of historical data
                        years = (max_date - min_date).days / 365.25
                        result['historical_depth'] = f"{years:.1f} years"
                    except:
                        pass
            
            print(f"   ✅ Available: {result['available']}")
            if result['columns']:
                print(f"   📊 Columns: {result['columns']}")
                print(f"   📈 Shape: {result['shape']}")
                print(f"   📅 Historical depth: {result['historical_depth']}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'available': False, 'error': str(e)}
    
    def _test_earnings_history(self, ticker, symbol: str) -> Dict[str, Any]:
        """Test ticker.earnings_history if available."""
        print("📋 Testing: ticker.earnings_history")
        try:
            if hasattr(ticker, 'earnings_history'):
                earnings_history = ticker.earnings_history
                result = {
                    'available': earnings_history is not None,
                    'data_type': str(type(earnings_history)),
                    'sample_data': str(earnings_history)[:200] if earnings_history else None
                }
            else:
                result = {'available': False, 'reason': 'Method not available'}
            
            print(f"   ✅ Available: {result['available']}")
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'available': False, 'error': str(e)}
    
    def _test_analyst_targets(self, ticker, symbol: str) -> Dict[str, Any]:
        """Test ticker.analyst_price_targets for revisions data."""
        print("🎯 Testing: ticker.analyst_price_targets")
        try:
            targets = ticker.analyst_price_targets
            result = {
                'available': targets is not None,
                'data_type': str(type(targets)),
                'sample_data': None
            }
            
            if targets is not None:
                if isinstance(targets, dict):
                    result['sample_data'] = targets
                elif hasattr(targets, 'to_dict'):
                    result['sample_data'] = targets.to_dict() if not targets.empty else "Empty DataFrame"
                else:
                    result['sample_data'] = str(targets)[:200]
            
            print(f"   ✅ Available: {result['available']}")
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'available': False, 'error': str(e)}
    
    def _test_recommendations(self, ticker, symbol: str) -> Dict[str, Any]:
        """Test ticker.recommendations for analyst sentiment."""
        print("💡 Testing: ticker.recommendations")
        try:
            recommendations = ticker.recommendations
            result = {
                'available': recommendations is not None and (not hasattr(recommendations, 'empty') or not recommendations.empty),
                'data_type': str(type(recommendations)),
                'shape': None,
                'columns': None,
                'sample_data': None,
                'historical_depth': None
            }
            
            if recommendations is not None and hasattr(recommendations, 'columns'):
                result['shape'] = recommendations.shape
                result['columns'] = list(recommendations.columns)
                result['sample_data'] = recommendations.head().to_dict() if not recommendations.empty else "Empty DataFrame"
                
                # Check historical depth
                if hasattr(recommendations.index, 'min'):
                    try:
                        min_date = recommendations.index.min()
                        max_date = recommendations.index.max()
                        years = (max_date - min_date).days / 365.25
                        result['historical_depth'] = f"{years:.1f} years"
                    except:
                        pass
            
            print(f"   ✅ Available: {result['available']}")
            if result['shape']:
                print(f"   📈 Shape: {result['shape']}")
                print(f"   📅 Historical depth: {result['historical_depth']}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'available': False, 'error': str(e)}
    
    def _test_earnings_estimate(self, ticker, symbol: str) -> Dict[str, Any]:
        """Test ticker.earnings_estimate if available."""
        print("🔮 Testing: ticker.earnings_estimate")
        try:
            if hasattr(ticker, 'earnings_estimate'):
                earnings_estimate = ticker.earnings_estimate
                result = {
                    'available': earnings_estimate is not None,
                    'data_type': str(type(earnings_estimate)),
                    'sample_data': str(earnings_estimate)[:200] if earnings_estimate else None
                }
            else:
                result = {'available': False, 'reason': 'Method not available'}
            
            print(f"   ✅ Available: {result['available']}")
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'available': False, 'error': str(e)}
    
    def _test_financials(self, ticker, symbol: str) -> Dict[str, Any]:
        """Test ticker.financials and related methods for earnings context."""
        print("💰 Testing: ticker.financials")
        try:
            financials = ticker.financials
            quarterly_financials = ticker.quarterly_financials
            
            result = {
                'financials_available': financials is not None and (not hasattr(financials, 'empty') or not financials.empty),
                'quarterly_financials_available': quarterly_financials is not None and (not hasattr(quarterly_financials, 'empty') or not quarterly_financials.empty),
                'financials_shape': financials.shape if financials is not None and hasattr(financials, 'shape') else None,
                'quarterly_financials_shape': quarterly_financials.shape if quarterly_financials is not None and hasattr(quarterly_financials, 'shape') else None,
                'has_earnings_metrics': False
            }
            
            # Check for earnings-related metrics in financials
            if financials is not None and hasattr(financials, 'index'):
                earnings_metrics = ['Net Income', 'Total Revenue', 'Basic EPS', 'Diluted EPS']
                available_metrics = [metric for metric in earnings_metrics if metric in financials.index]
                result['available_earnings_metrics'] = available_metrics
                result['has_earnings_metrics'] = len(available_metrics) > 0
            
            print(f"   ✅ Financials available: {result['financials_available']}")
            print(f"   ✅ Quarterly financials available: {result['quarterly_financials_available']}")
            if result['has_earnings_metrics']:
                print(f"   📊 Earnings metrics: {result['available_earnings_metrics']}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'available': False, 'error': str(e)}
    
    def _generate_summary(self):
        """Generate comprehensive summary of findings."""
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE YFINANCE EARNINGS CAPABILITIES SUMMARY")
        print("=" * 60)
        
        # Aggregate results across all symbols
        capabilities = {}
        for symbol, data in self.results.items():
            for method, result in data.items():
                if method not in capabilities:
                    capabilities[method] = {'available_count': 0, 'total_tested': 0, 'examples': []}
                
                capabilities[method]['total_tested'] += 1
                if result.get('available', False):
                    capabilities[method]['available_count'] += 1
                    capabilities[method]['examples'].append(symbol)
        
        print("\n🎯 METHOD AVAILABILITY ACROSS TEST SYMBOLS:")
        print("-" * 50)
        
        for method, info in capabilities.items():
            availability = info['available_count'] / info['total_tested']
            status = "✅" if availability >= 0.8 else "⚠️" if availability >= 0.5 else "❌"
            
            print(f"{status} {method:25} | {info['available_count']}/{info['total_tested']} symbols ({availability:.0%})")
            if info['examples']:
                print(f"    Examples: {', '.join(info['examples'][:3])}")
        
        # Historical depth analysis
        print("\n📅 HISTORICAL DATA DEPTH ANALYSIS:")
        print("-" * 40)
        
        for symbol, data in self.results.items():
            print(f"\n🔍 {symbol}:")
            
            # Check earnings_dates depth
            if data.get('earnings_dates', {}).get('historical_depth'):
                print(f"   📅 Earnings dates: {data['earnings_dates']['historical_depth']}")
            
            # Check quarterly_earnings depth
            if data.get('quarterly_earnings', {}).get('date_range'):
                print(f"   📈 Quarterly earnings: {data['quarterly_earnings']['date_range']}")
            
            # Check recommendations depth
            if data.get('recommendations', {}).get('historical_depth'):
                print(f"   💡 Recommendations: {data['recommendations']['historical_depth']}")
        
        # Generate recommendations
        print("\n🚀 RECOMMENDATIONS FOR EARNINGS ANALYSIS:")
        print("-" * 45)
        
        recommendations = []
        
        # Check if earnings_dates is consistently available
        earnings_dates_availability = capabilities.get('earnings_dates', {}).get('available_count', 0) / len(self.symbols)
        if earnings_dates_availability >= 0.8:
            recommendations.append("✅ Use ticker.earnings_dates for historical announcement dates")
        
        # Check quarterly earnings availability
        quarterly_earnings_availability = capabilities.get('quarterly_earnings', {}).get('available_count', 0) / len(self.symbols)
        if quarterly_earnings_availability >= 0.8:
            recommendations.append("✅ Use ticker.quarterly_earnings for earnings surprises")
        
        # Check recommendations availability
        recommendations_availability = capabilities.get('recommendations', {}).get('available_count', 0) / len(self.symbols)
        if recommendations_availability >= 0.5:
            recommendations.append("⚠️ ticker.recommendations available for analyst sentiment (limited coverage)")
        
        for rec in recommendations:
            print(f"   {rec}")
        
        print(f"\n📊 Total methods tested: {len(capabilities)}")
        print(f"🎯 Most reliable methods: earnings_dates, quarterly_earnings, financials")

def main():
    """Main research function."""
    # Test with a diverse set of symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "META", "TSLA"]
    
    researcher = YFinanceEarningsResearcher(test_symbols)
    results = researcher.research_all_capabilities()
    
    # Save results for future reference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/Users/dmartinsen/pers-prog/ml-pred/yfinance_earnings_research_{timestamp}.json"
    
    try:
        # Convert results to JSON-serializable format
        serializable_results = {}
        for symbol, data in results.items():
            serializable_results[symbol] = {}
            for method, result in data.items():
                serializable_results[symbol][method] = str(result)  # Convert complex objects to strings
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️ Could not save results: {e}")

if __name__ == "__main__":
    main()