import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime, timedelta

from utils.logging_utils import get_logger

logger = get_logger()

class BacktestPerformanceAnalyzer:
    """
    Comprehensive performance analysis for backtesting results.
    Provides detailed analytics, visualizations, and comparisons.
    """
    
    def __init__(self, results_data: Dict[str, Any] = None, results_file: str = None):
        """
        Initialize performance analyzer
        
        Args:
            results_data: Backtest results dictionary
            results_file: Path to results JSON file
        """
        self.results_data = None
        self.trades_data = None
        
        if results_data:
            self.results_data = results_data
        elif results_file:
            self._load_results_from_file(results_file)
        else:
            raise ValueError("Either results_data or results_file must be provided")
        
        # Extract key metrics
        self._extract_key_metrics()
        
        logger.info("Performance analyzer initialized")
    
    def _load_results_from_file(self, results_file: str):
        """Load results from JSON file"""
        try:
            results_path = Path(results_file)
            
            with open(results_path, 'r') as f:
                self.results_data = json.load(f)
            
            # Try to load corresponding trades file
            trades_file = results_path.parent / results_path.name.replace('backtest_results', 'trades_log')
            if trades_file.exists():
                with open(trades_file, 'r') as f:
                    self.trades_data = json.load(f)
            
            logger.info(f"Loaded results from: {results_file}")
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise
    
    def _extract_key_metrics(self):
        """Extract key metrics for easy access"""
        self.summary = self.results_data.get('backtest_summary', {})
        self.performance = self.results_data.get('trading_performance', {})
        self.risk_metrics = self.results_data.get('risk_metrics', {})
        self.trade_analysis = self.results_data.get('trade_analysis', {})
        
        # Calculate additional derived metrics
        self._calculate_derived_metrics()
    
    def _calculate_derived_metrics(self):
        """Calculate additional derived metrics"""
        try:
            # Monthly return rate
            if 'data_period' in self.results_data:
                start_date = pd.to_datetime(self.results_data['data_period']['start'])
                end_date = pd.to_datetime(self.results_data['data_period']['end'])
                period_months = (end_date - start_date).days / 30.44  # Average days per month
                
                self.monthly_return_rate = (self.summary.get('total_return_pct', 0) / period_months) if period_months > 0 else 0
                self.annualized_return = self.monthly_return_rate * 12
            else:
                self.monthly_return_rate = 0
                self.annualized_return = 0
            
            # Trade frequency metrics
            total_trades = self.performance.get('total_trades', 0)
            if 'data_period' in self.results_data and total_trades > 0:
                start_date = pd.to_datetime(self.results_data['data_period']['start'])
                end_date = pd.to_datetime(self.results_data['data_period']['end'])
                period_days = (end_date - start_date).days
                
                self.trades_per_day = total_trades / period_days if period_days > 0 else 0
                self.trades_per_week = self.trades_per_day * 7
            else:
                self.trades_per_day = 0
                self.trades_per_week = 0
            
        except Exception as e:
            logger.warning(f"Error calculating derived metrics: {e}")
            self.monthly_return_rate = 0
            self.annualized_return = 0
            self.trades_per_day = 0
            self.trades_per_week = 0
    
    def generate_performance_report(self) -> str:
        """
        Generate comprehensive performance report
        
        Returns:
            Formatted performance report string
        """
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("🎯 BACKTESTING PERFORMANCE REPORT")
        report.append("=" * 80)
        
        # Strategy Summary
        report.append(f"\n📋 STRATEGY SUMMARY")
        report.append(f"Hypothesis: {self.summary.get('hypothesis_name', 'Unknown')}")
        report.append(f"Model Type: {self.summary.get('model_type', 'Unknown')}")
        if 'data_period' in self.results_data:
            report.append(f"Period: {self.results_data['data_period']['start']} to {self.results_data['data_period']['end']}")
        report.append(f"Execution Time: {self.summary.get('execution_time_seconds', 0):.2f} seconds")
        
        # Financial Performance
        report.append(f"\n💰 FINANCIAL PERFORMANCE")
        report.append(f"Initial Capital: ${self.summary.get('initial_capital', 0):,.2f}")
        report.append(f"Final Equity: ${self.summary.get('final_equity', 0):,.2f}")
        report.append(f"Total Return: {self.summary.get('total_return_pct', 0):.2f}%")
        report.append(f"Monthly Return Rate: {self.monthly_return_rate:.2f}%")
        report.append(f"Annualized Return: {self.annualized_return:.2f}%")
        
        # Trading Statistics
        report.append(f"\n📊 TRADING STATISTICS")
        report.append(f"Total Trades: {self.performance.get('total_trades', 0)}")
        report.append(f"Winning Trades: {self.performance.get('winning_trades', 0)}")
        report.append(f"Losing Trades: {self.performance.get('losing_trades', 0)}")
        report.append(f"Win Rate: {self.performance.get('win_rate', 0):.1f}%")
        report.append(f"Trade Frequency: {self.trades_per_day:.2f} trades/day ({self.trades_per_week:.1f}/week)")
        
        # Risk Metrics
        report.append(f"\n⚠️  RISK METRICS")
        report.append(f"Maximum Drawdown: {self.performance.get('max_drawdown_pct', 0):.2f}%")
        report.append(f"Sharpe Ratio: {self.risk_metrics.get('sharpe_ratio', 0):.3f}")
        report.append(f"Sortino Ratio: {self.risk_metrics.get('sortino_ratio', 0):.3f}")
        report.append(f"Profit Factor: {self.risk_metrics.get('profit_factor', 0):.3f}")
        
        # Trade Analysis
        if self.trade_analysis and 'average_holding_period_minutes' in self.trade_analysis:
            report.append(f"\n📈 TRADE ANALYSIS")
            avg_hold = self.trade_analysis.get('average_holding_period_minutes', 0)
            report.append(f"Average Holding Period: {avg_hold:.1f} minutes ({avg_hold/60:.2f} hours)")
            report.append(f"Average P&L per Trade: ${self.trade_analysis.get('average_pnl_per_trade', 0):.2f}")
            report.append(f"Largest Win: {self.risk_metrics.get('largest_win', 0)*100:.2f}%")
            report.append(f"Largest Loss: {self.risk_metrics.get('largest_loss', 0)*100:.2f}%")
            report.append(f"Max Consecutive Wins: {self.trade_analysis.get('consecutive_wins_max', 0)}")
            report.append(f"Max Consecutive Losses: {self.trade_analysis.get('consecutive_losses_max', 0)}")
        
        # Exit Reasons
        if self.trade_analysis and 'exit_reasons' in self.trade_analysis:
            report.append(f"\n🚪 EXIT REASONS")
            for reason, count in self.trade_analysis['exit_reasons'].items():
                pct = (count / self.performance.get('total_trades', 1)) * 100
                report.append(f"{reason.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
        
        # Costs Analysis
        report.append(f"\n💸 COSTS ANALYSIS")
        total_fees = self.performance.get('total_fees', 0)
        total_return = self.summary.get('final_equity', 0) - self.summary.get('initial_capital', 0)
        cost_impact = (total_fees / abs(total_return)) * 100 if total_return != 0 else 0
        report.append(f"Total Transaction Fees: ${total_fees:.2f}")
        report.append(f"Cost Impact on Returns: {cost_impact:.2f}%")
        
        # Performance Rating
        report.append(f"\n🏆 PERFORMANCE RATING")
        rating = self._calculate_performance_rating()
        report.append(f"Overall Rating: {rating['rating']} ({rating['score']:.1f}/10)")
        report.append(f"Rating Factors: {rating['description']}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def _calculate_performance_rating(self) -> Dict[str, Any]:
        """Calculate overall performance rating"""
        score = 0
        factors = []
        
        # Return performance (30% weight)
        return_pct = self.summary.get('total_return_pct', 0)
        if return_pct > 20:
            score += 3
            factors.append("Excellent returns")
        elif return_pct > 10:
            score += 2.5
            factors.append("Good returns")
        elif return_pct > 5:
            score += 2
            factors.append("Moderate returns")
        elif return_pct > 0:
            score += 1
            factors.append("Positive returns")
        else:
            factors.append("Negative returns")
        
        # Sharpe ratio (25% weight)
        sharpe = self.risk_metrics.get('sharpe_ratio', 0)
        if sharpe > 2:
            score += 2.5
            factors.append("Excellent risk-adjusted returns")
        elif sharpe > 1:
            score += 2
            factors.append("Good risk-adjusted returns")
        elif sharpe > 0.5:
            score += 1.5
            factors.append("Moderate risk-adjusted returns")
        elif sharpe > 0:
            score += 1
            factors.append("Low risk-adjusted returns")
        
        # Drawdown (20% weight)
        drawdown = self.performance.get('max_drawdown_pct', 0)
        if drawdown < 5:
            score += 2
            factors.append("Low drawdown")
        elif drawdown < 10:
            score += 1.5
            factors.append("Moderate drawdown")
        elif drawdown < 20:
            score += 1
            factors.append("High drawdown")
        else:
            factors.append("Very high drawdown")
        
        # Win rate (15% weight)
        win_rate = self.performance.get('win_rate', 0)
        if win_rate > 70:
            score += 1.5
            factors.append("High win rate")
        elif win_rate > 50:
            score += 1
            factors.append("Good win rate")
        elif win_rate > 30:
            score += 0.5
            factors.append("Low win rate")
        
        # Trade frequency (10% weight)
        if self.trades_per_day > 5:
            score += 1
            factors.append("Active trading")
        elif self.trades_per_day > 1:
            score += 0.5
            factors.append("Moderate trading")
        
        # Rating scale
        if score >= 8.5:
            rating = "⭐⭐⭐⭐⭐ Excellent"
        elif score >= 7:
            rating = "⭐⭐⭐⭐ Very Good"
        elif score >= 5.5:
            rating = "⭐⭐⭐ Good"
        elif score >= 4:
            rating = "⭐⭐ Fair"
        elif score >= 2:
            rating = "⭐ Poor"
        else:
            rating = "☆ Very Poor"
        
        return {
            'score': score,
            'rating': rating,
            'description': ", ".join(factors)
        }
    
    def create_equity_curve(self, save_path: str = None) -> Optional[str]:
        """
        Create equity curve visualization
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None
        """
        try:
            if not self.trades_data or 'closed_trades' not in self.trades_data:
                logger.warning("No trade data available for equity curve")
                return None
            
            trades = self.trades_data['closed_trades']
            if not trades:
                logger.warning("No closed trades for equity curve")
                return None
            
            # Create equity curve data
            equity_data = []
            current_equity = self.summary.get('initial_capital', 10000)
            
            for trade in trades:
                equity_data.append({
                    'timestamp': pd.to_datetime(trade['exit_timestamp']),
                    'equity': current_equity + trade['net_pnl'],
                    'trade_pnl': trade['net_pnl'],
                    'cumulative_return_pct': ((current_equity + trade['net_pnl']) / self.summary.get('initial_capital', 10000) - 1) * 100
                })
                current_equity += trade['net_pnl']
            
            df_equity = pd.DataFrame(equity_data)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Main equity curve
            plt.subplot(2, 1, 1)
            plt.plot(df_equity['timestamp'], df_equity['equity'], linewidth=2, color='blue')
            plt.axhline(y=self.summary.get('initial_capital', 10000), color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
            plt.title(f"Equity Curve - {self.summary.get('hypothesis_name', 'Unknown Strategy')}")
            plt.ylabel('Equity ($)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Cumulative return percentage
            plt.subplot(2, 1, 2)
            plt.plot(df_equity['timestamp'], df_equity['cumulative_return_pct'], linewidth=2, color='green')
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            plt.title('Cumulative Return (%)')
            plt.xlabel('Date')
            plt.ylabel('Return (%)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Equity curve saved to: {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return None
            
        except Exception as e:
            logger.error(f"Error creating equity curve: {e}")
            return None
    
    def analyze_monthly_performance(self) -> pd.DataFrame:
        """
        Analyze monthly performance breakdown
        
        Returns:
            DataFrame with monthly performance metrics
        """
        try:
            if not self.trades_data or 'closed_trades' not in self.trades_data:
                logger.warning("No trade data available for monthly analysis")
                return pd.DataFrame()
            
            trades = self.trades_data['closed_trades']
            if not trades:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_trades = pd.DataFrame(trades)
            df_trades['exit_timestamp'] = pd.to_datetime(df_trades['exit_timestamp'])
            df_trades['month'] = df_trades['exit_timestamp'].dt.to_period('M')
            
            # Monthly aggregation
            monthly_stats = df_trades.groupby('month').agg({
                'net_pnl': ['sum', 'count', 'mean'],
                'net_pnl': lambda x: (x > 0).sum()  # Winning trades
            }).round(2)
            
            monthly_stats.columns = ['Total_PnL', 'Trade_Count', 'Avg_PnL', 'Winning_Trades']
            monthly_stats['Win_Rate'] = (monthly_stats['Winning_Trades'] / monthly_stats['Trade_Count'] * 100).round(1)
            monthly_stats['Monthly_Return_Pct'] = (monthly_stats['Total_PnL'] / self.summary.get('initial_capital', 10000) * 100).round(2)
            
            return monthly_stats
            
        except Exception as e:
            logger.error(f"Error analyzing monthly performance: {e}")
            return pd.DataFrame()
    
    def compare_to_benchmark(self, benchmark_return_pct: float = 5.0) -> Dict[str, Any]:
        """
        Compare strategy performance to benchmark
        
        Args:
            benchmark_return_pct: Benchmark annual return percentage
            
        Returns:
            Comparison metrics dictionary
        """
        strategy_return = self.annualized_return
        excess_return = strategy_return - benchmark_return_pct
        
        comparison = {
            'strategy_annual_return': strategy_return,
            'benchmark_annual_return': benchmark_return_pct,
            'excess_return': excess_return,
            'outperformed_benchmark': excess_return > 0,
            'sharpe_vs_benchmark': {
                'strategy_sharpe': self.risk_metrics.get('sharpe_ratio', 0),
                'benchmark_sharpe': benchmark_return_pct / 15,  # Assume 15% benchmark volatility
                'sharpe_advantage': self.risk_metrics.get('sharpe_ratio', 0) - (benchmark_return_pct / 15)
            }
        }
        
        return comparison
    
    def export_detailed_analysis(self, output_dir: str) -> Dict[str, str]:
        """
        Export comprehensive analysis to files
        
        Args:
            output_dir: Directory to save analysis files
            
        Returns:
            Dictionary with paths to created files
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            files_created = {}
            
            # Performance report
            report_file = output_path / f"performance_report_{timestamp}.txt"
            with open(report_file, 'w') as f:
                f.write(self.generate_performance_report())
            files_created['performance_report'] = str(report_file)
            
            # Equity curve
            equity_plot = output_path / f"equity_curve_{timestamp}.png"
            self.create_equity_curve(str(equity_plot))
            files_created['equity_curve'] = str(equity_plot)
            
            # Monthly analysis
            monthly_analysis = self.analyze_monthly_performance()
            if not monthly_analysis.empty:
                monthly_file = output_path / f"monthly_analysis_{timestamp}.csv"
                monthly_analysis.to_csv(monthly_file)
                files_created['monthly_analysis'] = str(monthly_file)
            
            # Detailed metrics JSON
            metrics_file = output_path / f"detailed_metrics_{timestamp}.json"
            detailed_metrics = {
                'summary_metrics': {
                    'total_return_pct': self.summary.get('total_return_pct', 0),
                    'annualized_return': self.annualized_return,
                    'sharpe_ratio': self.risk_metrics.get('sharpe_ratio', 0),
                    'max_drawdown_pct': self.performance.get('max_drawdown_pct', 0),
                    'win_rate': self.performance.get('win_rate', 0),
                    'total_trades': self.performance.get('total_trades', 0),
                    'profit_factor': self.risk_metrics.get('profit_factor', 0)
                },
                'derived_metrics': {
                    'monthly_return_rate': self.monthly_return_rate,
                    'trades_per_day': self.trades_per_day,
                    'trades_per_week': self.trades_per_week
                },
                'performance_rating': self._calculate_performance_rating(),
                'benchmark_comparison': self.compare_to_benchmark()
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(detailed_metrics, f, indent=2, default=str)
            files_created['detailed_metrics'] = str(metrics_file)
            
            logger.info(f"Detailed analysis exported to: {output_dir}")
            return files_created
            
        except Exception as e:
            logger.error(f"Error exporting analysis: {e}")
            return {}
    
    def get_key_insights(self) -> List[str]:
        """
        Generate key insights from the analysis
        
        Returns:
            List of key insight strings
        """
        insights = []
        
        # Performance insights
        return_pct = self.summary.get('total_return_pct', 0)
        if return_pct > 15:
            insights.append(f"🎯 Strong performance with {return_pct:.1f}% total return")
        elif return_pct > 0:
            insights.append(f"✅ Positive returns achieved: {return_pct:.1f}%")
        else:
            insights.append(f"⚠️ Strategy experienced losses: {return_pct:.1f}%")
        
        # Risk insights
        sharpe = self.risk_metrics.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            insights.append(f"📈 Excellent risk-adjusted returns (Sharpe: {sharpe:.2f})")
        elif sharpe > 1:
            insights.append(f"👍 Good risk-adjusted returns (Sharpe: {sharpe:.2f})")
        elif sharpe > 0:
            insights.append(f"📊 Moderate risk-adjusted returns (Sharpe: {sharpe:.2f})")
        else:
            insights.append(f"📉 Poor risk-adjusted returns (Sharpe: {sharpe:.2f})")
        
        # Trading frequency insights
        if self.trades_per_day > 5:
            insights.append(f"⚡ High-frequency strategy: {self.trades_per_day:.1f} trades/day")
        elif self.trades_per_day > 1:
            insights.append(f"📊 Active trading strategy: {self.trades_per_day:.1f} trades/day")
        else:
            insights.append(f"🐌 Low-frequency strategy: {self.trades_per_day:.1f} trades/day")
        
        # Win rate insights
        win_rate = self.performance.get('win_rate', 0)
        if win_rate > 60:
            insights.append(f"🎯 High accuracy strategy with {win_rate:.1f}% win rate")
        elif win_rate > 40:
            insights.append(f"⚖️ Balanced win rate: {win_rate:.1f}%")
        else:
            insights.append(f"🔄 Lower win rate ({win_rate:.1f}%) but may have larger average wins")
        
        # Drawdown insights
        drawdown = self.performance.get('max_drawdown_pct', 0)
        if drawdown < 5:
            insights.append(f"🛡️ Low risk profile with {drawdown:.1f}% max drawdown")
        elif drawdown < 15:
            insights.append(f"⚠️ Moderate risk with {drawdown:.1f}% max drawdown")
        else:
            insights.append(f"⚡ High risk strategy with {drawdown:.1f}% max drawdown")
        
        return insights