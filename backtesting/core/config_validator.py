from typing import Dict, Any, List, Tuple
from pathlib import Path
from utils.logging_utils import get_logger

logger = get_logger()

class BacktestConfigValidator:
    """
    Validates backtesting configuration and provides clear error messages.
    Ensures all required parameters are present and within valid ranges.
    """
    
    def __init__(self):
        """Initialize config validator"""
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate complete backtesting configuration
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_messages, warning_messages)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Check top-level structure
        self._validate_structure(config)
        
        # Validate each section
        if 'backtesting' in config:
            backtest_config = config['backtesting']
            self._validate_basic_config(backtest_config)
            self._validate_data_config(backtest_config)
            self._validate_capital_config(backtest_config)
            self._validate_execution_config(backtest_config)
            self._validate_signal_processing_config(backtest_config)
            self._validate_risk_management_config(backtest_config)
            self._validate_position_management_config(backtest_config)
        
        # Validate hypothesis-specific requirements using plugin system
        self._validate_hypothesis_config(config)
        
        is_valid = len(self.validation_errors) == 0
        
        if is_valid:
            logger.info("Configuration validation passed")
        else:
            logger.error(f"Configuration validation failed with {len(self.validation_errors)} errors")
        
        if self.validation_warnings:
            logger.warning(f"Configuration has {len(self.validation_warnings)} warnings")
        
        return is_valid, self.validation_errors, self.validation_warnings
    
    def _validate_structure(self, config: Dict[str, Any]):
        """Validate top-level configuration structure"""
        required_sections = ['backtesting']
        
        for section in required_sections:
            if section not in config:
                self.validation_errors.append(f"Missing required configuration section: '{section}'")
        
        if 'hypothesis_name' not in config:
            self.validation_warnings.append("Missing 'hypothesis_name' - using 'unknown'")
    
    def _validate_basic_config(self, config: Dict[str, Any]):
        """Validate basic backtesting configuration"""
        # Required boolean fields
        if 'enabled' not in config:
            self.validation_errors.append("Missing required field: backtesting.enabled")
        elif not isinstance(config['enabled'], bool):
            self.validation_errors.append("Field 'backtesting.enabled' must be boolean")
        
        # Model folder validation
        model_folder = config.get('model_folder')
        if model_folder and not Path(model_folder).exists():
            self.validation_errors.append(f"Model folder does not exist: {model_folder}")
    
    def _validate_data_config(self, config: Dict[str, Any]):
        """Validate data configuration"""
        data_source = config.get('data_source')
        if data_source:
            data_path = Path(data_source)
            if not data_path.exists():
                self.validation_errors.append(f"Data source file does not exist: {data_source}")
            elif not data_path.suffix == '.csv':
                self.validation_warnings.append(f"Data source is not a CSV file: {data_source}")
        
        # Date validation
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        
        if start_date and end_date:
            try:
                import pandas as pd
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                if start_dt >= end_dt:
                    self.validation_errors.append("start_date must be before end_date")
                
                # Check if date range is reasonable
                date_diff = (end_dt - start_dt).days
                if date_diff > 365:
                    self.validation_warnings.append("Date range > 1 year may result in slow backtesting")
                elif date_diff < 1:
                    self.validation_warnings.append("Date range < 1 day may not provide sufficient data")
                    
            except Exception as e:
                self.validation_errors.append(f"Invalid date format in start_date or end_date: {e}")
        
        # Buffer size validation
        buffer_size = config.get('buffer_size')
        if buffer_size is not None:
            if not isinstance(buffer_size, int) or buffer_size < 10:
                self.validation_errors.append("buffer_size must be integer >= 10")
            elif buffer_size > 1000:
                self.validation_warnings.append("Large buffer_size (>1000) may consume excessive memory")
    
    def _validate_capital_config(self, config: Dict[str, Any]):
        """Validate capital and cost configuration"""
        # Initial capital
        initial_capital = config.get('initial_capital')
        if initial_capital is None:
            self.validation_errors.append("Missing required field: initial_capital")
        else:
            try:
                capital = float(initial_capital)
                if capital < 100:
                    self.validation_errors.append("initial_capital must be >= $100")
                elif capital > 10000000:  # $10M
                    self.validation_warnings.append("initial_capital > $10M - ensure this is intended")
            except (ValueError, TypeError):
                self.validation_errors.append("initial_capital must be numeric")
        
        # Transaction costs
        transaction_costs = config.get('transaction_costs')
        if transaction_costs is not None:
            try:
                cost = float(transaction_costs)
                if not (0.0 <= cost <= 0.01):  # 0% to 1%
                    self.validation_errors.append("transaction_costs must be between 0.0 and 0.01 (0-1%)")
            except (ValueError, TypeError):
                self.validation_errors.append("transaction_costs must be numeric")
        
        # Slippage
        slippage = config.get('slippage')
        if slippage is not None:
            try:
                slip = float(slippage)
                if not (0.0 <= slip <= 0.01):  # 0% to 1%
                    self.validation_errors.append("slippage must be between 0.0 and 0.01 (0-1%)")
            except (ValueError, TypeError):
                self.validation_errors.append("slippage must be numeric")
    
    def _validate_execution_config(self, config: Dict[str, Any]):
        """Validate execution configuration"""
        execution_config = config.get('execution', {})
        
        # Max loops
        max_loops = execution_config.get('max_loops')
        if max_loops is not None:
            try:
                loops = int(max_loops)
                if loops < 10:
                    self.validation_warnings.append("max_loops < 10 may not provide sufficient backtesting")
                elif loops > 100000:
                    self.validation_warnings.append("max_loops > 100k may result in very slow backtesting")
            except (ValueError, TypeError):
                self.validation_errors.append("execution.max_loops must be integer")
    
    def _validate_signal_processing_config(self, config: Dict[str, Any]):
        """Validate signal processing configuration"""
        signal_config = config.get('signal_processing', {})
        
        # Confidence threshold
        confidence_threshold = signal_config.get('confidence_threshold')
        if confidence_threshold is not None:
            try:
                conf = float(confidence_threshold)
                if not (0.0 <= conf <= 1.0):
                    self.validation_errors.append("confidence_threshold must be between 0.0 and 1.0")
                elif conf < 0.3:
                    self.validation_warnings.append("confidence_threshold < 0.3 may generate many false signals")
                elif conf > 0.9:
                    self.validation_warnings.append("confidence_threshold > 0.9 may miss valid opportunities")
            except (ValueError, TypeError):
                self.validation_errors.append("confidence_threshold must be numeric")
        
        # Processor type
        processor_type = signal_config.get('processor_type')
        if processor_type and processor_type not in ['basic', 'advanced', 'smart', 'ultra']:
            self.validation_errors.append("processor_type must be one of: basic, advanced, smart, ultra")
        
        # Signal filters
        signal_filters = signal_config.get('signal_filters', [])
        if not isinstance(signal_filters, list):
            self.validation_errors.append("signal_filters must be a list")
        else:
            valid_filters = ['session_filter', 'volatility_filter', 'volume_filter']
            for filter_name in signal_filters:
                if filter_name not in valid_filters:
                    self.validation_warnings.append(f"Unknown signal filter: {filter_name}")
    
    def _validate_risk_management_config(self, config: Dict[str, Any]):
        """Validate risk management configuration"""
        risk_config = config.get('risk_management', {})
        
        # Position size
        max_position_size = risk_config.get('max_position_size')
        if max_position_size is not None:
            try:
                size = float(max_position_size)
                if not (0.001 <= size <= 0.2):  # 0.1% to 20%
                    self.validation_errors.append("max_position_size must be between 0.001 and 0.2 (0.1%-20%)")
                elif size > 0.05:
                    self.validation_warnings.append("max_position_size > 5% is high risk")
            except (ValueError, TypeError):
                self.validation_errors.append("max_position_size must be numeric")
        
        # Stop loss
        stop_loss_pct = risk_config.get('stop_loss_pct')
        if stop_loss_pct is not None:
            try:
                stop = float(stop_loss_pct)
                if not (0.001 <= stop <= 0.1):  # 0.1% to 10%
                    self.validation_errors.append("stop_loss_pct must be between 0.001 and 0.1 (0.1%-10%)")
            except (ValueError, TypeError):
                self.validation_errors.append("stop_loss_pct must be numeric")
        
        # Take profit
        take_profit_pct = risk_config.get('take_profit_pct')
        if take_profit_pct is not None:
            try:
                profit = float(take_profit_pct)
                if not (0.001 <= profit <= 0.2):  # 0.1% to 20%
                    self.validation_errors.append("take_profit_pct must be between 0.001 and 0.2 (0.1%-20%)")
            except (ValueError, TypeError):
                self.validation_errors.append("take_profit_pct must be numeric")
        
        # Risk/reward ratio check
        if stop_loss_pct and take_profit_pct:
            try:
                stop = float(stop_loss_pct)
                profit = float(take_profit_pct)
                ratio = profit / stop
                if ratio < 1.0:
                    self.validation_warnings.append(f"Risk/reward ratio {ratio:.2f} < 1.0 may not be profitable")
            except:
                pass
        
        # Max drawdown
        max_drawdown = risk_config.get('max_drawdown')
        if max_drawdown is not None:
            try:
                drawdown = float(max_drawdown)
                if not (0.01 <= drawdown <= 0.5):  # 1% to 50%
                    self.validation_errors.append("max_drawdown must be between 0.01 and 0.5 (1%-50%)")
                elif drawdown > 0.2:
                    self.validation_warnings.append("max_drawdown > 20% is very high risk")
            except (ValueError, TypeError):
                self.validation_errors.append("max_drawdown must be numeric")
        
        # Concurrent positions
        max_concurrent = risk_config.get('max_concurrent_positions')
        if max_concurrent is not None:
            try:
                positions = int(max_concurrent)
                if not (1 <= positions <= 20):
                    self.validation_errors.append("max_concurrent_positions must be between 1 and 20")
                elif positions > 10:
                    self.validation_warnings.append("max_concurrent_positions > 10 may be difficult to manage")
            except (ValueError, TypeError):
                self.validation_errors.append("max_concurrent_positions must be integer")
    
    def _validate_position_management_config(self, config: Dict[str, Any]):
        """Validate position management configuration"""
        position_config = config.get('position_management', {})
        
        # Position tracking
        tracking = position_config.get('position_tracking')
        if tracking and tracking not in ['basic', 'detailed']:
            self.validation_errors.append("position_tracking must be 'basic' or 'detailed'")
        
        # Consistency check with risk management
        risk_concurrent = config.get('risk_management', {}).get('max_concurrent_positions')
        pos_concurrent = position_config.get('max_concurrent_positions')
        
        if risk_concurrent and pos_concurrent and risk_concurrent != pos_concurrent:
            self.validation_warnings.append("max_concurrent_positions differs between risk_management and position_management sections")
    
    def _validate_hypothesis_config(self, config: Dict[str, Any]):
        """Validate hypothesis-specific configuration using plugin system"""
        try:
            from ..hypothesis_plugins import HypothesisPluginRegistry
            
            hypothesis_name = config.get('hypothesis_name', 'unknown')
            is_valid, plugin_errors = HypothesisPluginRegistry.validate_plugin_config(hypothesis_name, config)
            
            if not is_valid:
                self.validation_errors.extend(plugin_errors)
                
        except ImportError:
            self.validation_warnings.append("Could not validate hypothesis-specific configuration - plugin system not available")
        except Exception as e:
            self.validation_warnings.append(f"Hypothesis validation warning: {e}")
    
    def get_config_summary(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of the configuration for review
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Summary dictionary
        """
        backtest_config = config.get('backtesting', {})
        
        summary = {
            'hypothesis_name': config.get('hypothesis_name', 'unknown'),
            'capital': backtest_config.get('initial_capital', 'not set'),
            'data_source': Path(backtest_config.get('data_source', '')).name if backtest_config.get('data_source') else 'not set',
            'date_range': f"{backtest_config.get('start_date', 'not set')} to {backtest_config.get('end_date', 'not set')}",
            'max_position_size': f"{backtest_config.get('risk_management', {}).get('max_position_size', 'not set')}",
            'stop_loss': f"{backtest_config.get('risk_management', {}).get('stop_loss_pct', 'not set')}",
            'confidence_threshold': f"{backtest_config.get('signal_processing', {}).get('confidence_threshold', 'not set')}",
            'processor_type': backtest_config.get('signal_processing', {}).get('processor_type', 'basic')
        }
        
        return summary