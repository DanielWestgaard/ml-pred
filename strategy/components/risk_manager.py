import numpy as np
from typing import Dict, List, Any, Optional
from utils.logging_utils import get_logger

logger = get_logger()

class RiskManager:
    """
    Manages risk for trading strategy including position sizing,
    stop losses, and portfolio-level risk controls.
    """
    
    def __init__(self, max_position_size: float = 0.02, stop_loss_pct: float = 0.01,
                 take_profit_pct: float = 0.02, max_drawdown: float = 0.05,
                 max_concurrent_positions: int = 3, **kwargs):
        """
        Initialize risk manager
        
        Args:
            max_position_size: Maximum position size as percentage of capital (e.g., 0.02 = 2%)
            stop_loss_pct: Stop loss percentage (e.g., 0.01 = 1%)
            take_profit_pct: Take profit percentage (e.g., 0.02 = 2%)
            max_drawdown: Maximum allowable drawdown (e.g., 0.05 = 5%)
            max_concurrent_positions: Maximum number of concurrent positions
        """
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown = max_drawdown
        self.max_concurrent_positions = max_concurrent_positions
        
        # Risk tracking
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        
        logger.info(f"Risk manager initialized - Max position: {max_position_size*100}%, "
                    f"Stop loss: {stop_loss_pct*100}%, Max drawdown: {max_drawdown*100}%")
    
    def calculate_position_size(self, signal_strength: float, account_balance: float,
                              current_price: float, direction: str = 'BUY', volatility: float = None) -> Dict[str, Any]:
        """
        Calculate appropriate position size based on risk parameters
        
        Args:
            signal_strength: Strength of the trading signal (0-1)
            account_balance: Current account balance. TODO: Make dynamic with provider.
            current_price: Current market price
            direction: Trade direction ('BUY' or 'SELL')
            volatility: Current market volatility (optional)
            
        Returns:
            Dictionary with position sizing information
        """
        # Base position size
        base_size = account_balance * self.max_position_size
        
        # Adjust for signal strength
        strength_adjusted_size = base_size * signal_strength
        
        # Adjust for volatility if provided
        if volatility:
            # Reduce position size in high volatility
            volatility_adjustment = max(0.5, min(1.0, 1.0 - (volatility * 2)))
            strength_adjusted_size *= volatility_adjustment
        
        # Convert to units
        position_units = strength_adjusted_size / current_price
        
        # Calculate stop loss and take profit levels based on direction
        if direction == 'BUY':
            stop_loss_price = current_price * (1 - self.stop_loss_pct)
            take_profit_price = current_price * (1 + self.take_profit_pct)
        else:  # SELL
            stop_loss_price = current_price * (1 + self.stop_loss_pct)
            take_profit_price = current_price * (1 - self.take_profit_pct)
        
        return {
            'position_size_usd': strength_adjusted_size,
            'position_units': position_units,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'risk_per_unit': current_price * self.stop_loss_pct,
            'max_loss_usd': strength_adjusted_size * self.stop_loss_pct
        }
    
    def validate_trade(self, position_info: Dict[str, Any], current_positions: List[Dict],
                      account_balance: float) -> Dict[str, Any]:
        """
        Validate if a trade should be executed based on risk rules
        
        Args:
            position_info: Position information from calculate_position_size
            current_positions: List of current open positions
            account_balance: Current account balance
            
        Returns:
            Validation result with approval status and reasons
        """
        validation = {
            'approved': True,
            'reasons': [],
            'risk_score': 0.0
        }
        
        # Check maximum concurrent positions
        if len(current_positions) >= self.max_concurrent_positions:
            validation['approved'] = False
            validation['reasons'].append(f"Maximum concurrent positions ({self.max_concurrent_positions}) reached")
        
        # Check drawdown limits
        if self.current_drawdown >= self.max_drawdown:
            validation['approved'] = False
            validation['reasons'].append(f"Maximum drawdown ({self.max_drawdown*100}%) exceeded")
        
        # Check position size relative to account
        position_pct = position_info['position_size_usd'] / account_balance
        if position_pct > self.max_position_size:
            validation['approved'] = False
            validation['reasons'].append(f"Position size ({position_pct*100:.2f}%) exceeds maximum ({self.max_position_size*100}%)")
        
        # Calculate overall risk score
        portfolio_risk = sum([pos.get('max_loss_usd', 0) for pos in current_positions])
        total_risk = portfolio_risk + position_info['max_loss_usd']
        risk_pct = total_risk / account_balance
        validation['risk_score'] = risk_pct
        
        # Check if total portfolio risk is acceptable
        if risk_pct > self.max_position_size * 3:  # Conservative total risk limit
            validation['approved'] = False
            validation['reasons'].append(f"Total portfolio risk ({risk_pct*100:.2f}%) too high")
        
        if validation['approved']:
            logger.info(f"Trade validated - Risk score: {risk_pct*100:.2f}%")
        else:
            logger.warning(f"Trade rejected - Reasons: {', '.join(validation['reasons'])}")
        
        return validation
    
    def update_drawdown(self, current_equity: float):
        """
        Update current drawdown tracking
        
        Args:
            current_equity: Current account equity
        """
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        # Log significant drawdowns
        if self.current_drawdown > 0.02:  # 2% threshold
            logger.warning(f"Current drawdown: {self.current_drawdown*100:.2f}%")
    
    def should_close_position(self, position: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Determine if a position should be closed based on risk rules
        
        Args:
            position: Position information
            current_price: Current market price
            
        Returns:
            Dictionary with close recommendation and reason
        """
        close_info = {
            'should_close': False,
            'reason': None,
            'urgency': 'normal'
        }
        
        entry_price = position.get('entry_price', 0)
        direction = position.get('direction', 'BUY')
        stop_loss = position.get('stop_loss_price')
        take_profit = position.get('take_profit_price')
        
        if direction == 'BUY':
            # Check stop loss
            if stop_loss and current_price <= stop_loss:
                close_info.update({
                    'should_close': True,
                    'reason': 'stop_loss_hit',
                    'urgency': 'high'
                })
            # Check take profit
            elif take_profit and current_price >= take_profit:
                close_info.update({
                    'should_close': True,
                    'reason': 'take_profit_hit',
                    'urgency': 'normal'
                })
        
        elif direction == 'SELL':
            # Check stop loss
            if stop_loss and current_price >= stop_loss:
                close_info.update({
                    'should_close': True,
                    'reason': 'stop_loss_hit',
                    'urgency': 'high'
                })
            # Check take profit
            elif take_profit and current_price <= take_profit:
                close_info.update({
                    'should_close': True,
                    'reason': 'take_profit_hit',
                    'urgency': 'normal'
                })
        
        return close_info
    
    def can_open_position(self, signal: Dict[str, Any], position_manager=None) -> bool:
        """
        Check if a position can be opened based on the signal and risk management rules
        
        Args:
            signal: Trading signal dictionary containing confidence, direction, symbol, etc.
            position_manager: Position manager instance to get current positions
            
        Returns:
            Boolean indicating if position can be opened
        """
        try:
            # Extract signal information
            signal_strength = signal.get('confidence', 0.5)
            current_price = signal.get('current_price', 1.0)
            
            # Get actual position data if available
            if position_manager:
                current_positions = position_manager.get_open_positions()
                # Use placeholder account balance for now - should be integrated with broker
                account_balance = 10000.0
            else:
                # Fallback to placeholder values
                account_balance = 10000.0
                current_positions = []
            
            # Calculate position size
            position_info = self.calculate_position_size(
                signal_strength=signal_strength,
                account_balance=account_balance,
                current_price=current_price
            )
            
            # Validate the trade
            validation_result = self.validate_trade(
                position_info=position_info,
                current_positions=current_positions,
                account_balance=account_balance
            )
            
            return validation_result['approved']
            
        except Exception as e:
            logger.error(f"Error in can_open_position: {e}")
            return False

    def calculate_stop_loss(self, signal: Dict[str, Any]) -> float:
        """
        Calculate stop loss price for a signal
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Stop loss price
        """
        current_price = signal.get('current_price', 1.0)
        direction = signal.get('direction', 'BUY')
        
        if direction == 'BUY':
            return current_price * (1 - self.stop_loss_pct)
        else:  # SELL
            return current_price * (1 + self.stop_loss_pct)

    def calculate_take_profit(self, signal: Dict[str, Any]) -> float:
        """
        Calculate take profit price for a signal
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Take profit price
        """
        current_price = signal.get('current_price', 1.0)
        direction = signal.get('direction', 'BUY')
        
        if direction == 'BUY':
            return current_price * (1 + self.take_profit_pct)
        else:  # SELL
            return current_price * (1 - self.take_profit_pct)

    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get current risk summary
        
        Returns:
            Dictionary with risk metrics
        """
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown_limit': self.max_drawdown,
            'peak_equity': self.peak_equity,
            'max_position_size': self.max_position_size,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_concurrent_positions': self.max_concurrent_positions
        }