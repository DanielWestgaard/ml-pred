"""
Modernized Backtesting Orchestrator

This is the new, plugin-based backtesting orchestrator that leverages:
- Hypothesis plugin system for modularity
- Focused core components for maintainability  
- Existing feature generation system for consistency
- Comprehensive configuration validation

Usage:
    # Modern approach with plugins
    orchestrator = ModernizedBacktester(hypothesis_config_path="path/to/config.yml")
    results = orchestrator.run_backtest()
    
    # Legacy compatibility maintained
    from backtesting.backtester import BacktestOrchestrator  # Still works
"""

import time
from typing import Dict, Any, Optional
from pathlib import Path

from utils.logging_utils import get_logger
from backtesting.core import BacktestEngine

logger = get_logger()

class ModernizedBacktester:
    """
    Modern backtesting orchestrator using the plugin system.
    
    This class provides a clean interface to the new modular backtesting
    architecture while maintaining compatibility with existing code.
    """
    
    def __init__(self, hypothesis_config_path: str = None, model_folder: str = None, **kwargs):
        """
        Initialize modernized backtester
        
        Args:
            hypothesis_config_path: Path to hypothesis configuration file
            model_folder: Optional model folder path override
        """
        self.hypothesis_config_path = hypothesis_config_path
        self.model_folder = model_folder
        self.kwargs = kwargs
        
        # Core engine
        self.engine = None
        
        # Results storage
        self.backtest_results = None
        
        logger.info("ModernizedBacktester initialized")
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run complete backtesting simulation using the new architecture
        
        Returns:
            Comprehensive backtest results
        """
        try:
            logger.info("🚀 Starting modernized backtesting...")
            start_time = time.time()
            
            # Initialize the core engine
            self.engine = BacktestEngine(
                hypothesis_config_path=self.hypothesis_config_path,
                model_folder=self.model_folder,
                **self.kwargs
            )
            
            # Run the backtest
            self.backtest_results = self.engine.run_backtest()
            
            execution_time = time.time() - start_time
            logger.info(f"✅ Modernized backtest completed in {execution_time:.2f} seconds")
            
            return self.backtest_results
            
        except Exception as e:
            logger.error(f"Error in modernized backtesting: {e}")
            raise
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current backtest progress"""
        if self.engine:
            return self.engine.get_progress()
        else:
            return {'error': 'Backtest not started'}
    
    def stop(self):
        """Stop the backtest"""
        if self.engine:
            self.engine.stop()
        else:
            logger.warning("No active backtest to stop")
    
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get information about the components being used
        
        Returns:
            Dictionary with component information
        """
        if not self.engine:
            return {'error': 'Engine not initialized'}
        
        try:
            return {
                'model_info': self.engine.model_handler.get_model_info(),
                'signal_processor_stats': self.engine.signal_handler.get_processor_stats(),
                'exit_strategy_info': self.engine.signal_handler.get_exit_strategy_info(),
                'signal_tracker_info': self.engine.signal_handler.get_signal_tracker_info(),
                'hypothesis_plugin': self.engine.hypothesis_plugin.get_hypothesis_info(),
                'configuration_summary': self.engine.config_validator.get_config_summary(self.engine.config)
            }
        except Exception as e:
            logger.error(f"Error getting component info: {e}")
            return {'error': str(e)}
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate configuration without running backtest
        
        Returns:
            Validation results
        """
        try:
            # Temporarily initialize engine for validation
            temp_engine = BacktestEngine(
                hypothesis_config_path=self.hypothesis_config_path,
                model_folder=self.model_folder,
                **self.kwargs
            )
            
            # Just return validation results without running backtest
            return {
                'is_valid': True,
                'config_summary': temp_engine.config_validator.get_config_summary(temp_engine.config),
                'hypothesis_info': temp_engine.hypothesis_plugin.get_hypothesis_info(),
                'model_info': temp_engine.model_handler.get_model_info()
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': str(e),
                'message': 'Configuration validation failed'
            }
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a concise summary of backtest results
        
        Returns:
            Summary of key metrics
        """
        if not self.backtest_results:
            return {'error': 'No backtest results available'}
        
        summary = self.backtest_results.get('backtest_summary', {})
        performance = self.backtest_results.get('trading_performance', {})
        risk_metrics = self.backtest_results.get('risk_metrics', {})
        
        return {
            'hypothesis': summary.get('hypothesis_name', 'Unknown'),
            'total_return_pct': summary.get('total_return_pct', 0),
            'total_trades': performance.get('total_trades', 0),
            'win_rate': performance.get('win_rate', 0),
            'max_drawdown_pct': performance.get('max_drawdown_pct', 0),
            'sharpe_ratio': risk_metrics.get('sharpe_ratio', 'N/A'),
            'profit_factor': risk_metrics.get('profit_factor', 'N/A'),
            'execution_time': summary.get('execution_time_seconds', 0)
        }


# Convenience function for quick backtesting
def run_quick_backtest(hypothesis_config_path: str, model_folder: str = None) -> Dict[str, Any]:
    """
    Convenience function for running a quick backtest
    
    Args:
        hypothesis_config_path: Path to hypothesis configuration
        model_folder: Optional model folder override
        
    Returns:
        Backtest results
    """
    backtester = ModernizedBacktester(
        hypothesis_config_path=hypothesis_config_path,
        model_folder=model_folder
    )
    
    return backtester.run_backtest()


# Compatibility alias - maintain backward compatibility
def create_legacy_compatible_orchestrator(hypothesis_config_path: str = None, 
                                        model_folder: str = None, **kwargs):
    """
    Create a modernized backtester with legacy-compatible interface
    
    This function allows existing code to use the new architecture with minimal changes:
    
    Before:
        from backtesting.backtester import BacktestOrchestrator
        orchestrator = BacktestOrchestrator(hypothesis_config_path="config.yml")
        
    After:
        from backtesting.modernized_backtester import create_legacy_compatible_orchestrator as BacktestOrchestrator
        orchestrator = BacktestOrchestrator(hypothesis_config_path="config.yml")
    """
    return ModernizedBacktester(hypothesis_config_path, model_folder, **kwargs)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        model_folder = sys.argv[2] if len(sys.argv) > 2 else None
        
        logger.info(f"Running modernized backtest with config: {config_path}")
        
        try:
            results = run_quick_backtest(config_path, model_folder)
            print("\n" + "="*60)
            print("🎯 BACKTEST RESULTS SUMMARY")
            print("="*60)
            
            summary = results.get('backtest_summary', {})
            performance = results.get('trading_performance', {})
            
            print(f"Hypothesis: {summary.get('hypothesis_name', 'Unknown')}")
            print(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
            print(f"Total Trades: {performance.get('total_trades', 0)}")
            print(f"Win Rate: {performance.get('win_rate', 0):.1f}%")
            print(f"Max Drawdown: {performance.get('max_drawdown_pct', 0):.2f}%")
            print(f"Execution Time: {summary.get('execution_time_seconds', 0):.2f}s")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            sys.exit(1)
    else:
        print("Usage: python modernized_backtester.py <config_path> [model_folder]")
        sys.exit(1)