import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from utils.logging_utils import get_logger

logger = get_logger()

class SimulationBroker:
    """
    Simulation broker for backtesting that provides instant fills
    with realistic transaction costs and slippage.
    """
    
    def __init__(self, initial_capital: float = 10000.0, 
                 transaction_cost_pct: float = 0.0013,  # 1.3 pips for GBPUSD
                 slippage_pct: float = 0.0002,          # 0.2 pips average slippage
                 **kwargs):
        """
        Initialize simulation broker
        
        Args:
            initial_capital: Starting capital for backtesting
            transaction_cost_pct: Transaction cost as percentage (e.g., 0.0013 = 1.3 pips)
            slippage_pct: Average slippage as percentage (e.g., 0.0002 = 0.2 pips)
        """
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self.current_equity = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        
        # Trade tracking
        self.open_positions = []
        self.closed_trades = []
        self.trade_counter = 0
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees = 0.0
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        
        logger.info(f"Simulation broker initialized - Capital: ${initial_capital:,.2f}, "
                   f"Transaction cost: {transaction_cost_pct*100:.3f}%, Slippage: {slippage_pct*100:.3f}%")
    
    def place_order(self, signal: Dict[str, Any], current_price: float, 
                   position_size: float, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """
        Place an order in simulation (instant fill)
        
        Args:
            signal: Trading signal dictionary
            current_price: Current market price
            position_size: Position size in USD
            timestamp: Current timestamp
            
        Returns:
            Order execution result
        """
        try:
            direction = signal.get('direction', 'BUY')
            
            # Apply slippage (random within realistic range)
            slippage_factor = np.random.uniform(-self.slippage_pct, self.slippage_pct)
            if direction == 'BUY':
                execution_price = current_price * (1 + abs(slippage_factor))  # Buy at higher price
            else:
                execution_price = current_price * (1 - abs(slippage_factor))  # Sell at lower price
            
            # Calculate position units
            position_units = position_size / execution_price
            
            # Calculate transaction costs
            transaction_cost = position_size * self.transaction_cost_pct
            self.total_fees += transaction_cost
            
            # Create position
            self.trade_counter += 1
            position = {
                'id': f"trade_{self.trade_counter}",
                'symbol': signal.get('symbol', 'GBPUSD'),
                'direction': direction,
                'entry_price': execution_price,
                'position_size_usd': position_size,
                'position_units': position_units,
                'entry_timestamp': timestamp,
                'stop_loss_price': signal.get('stop_loss_price'),
                'take_profit_price': signal.get('take_profit_price'),
                'transaction_cost': transaction_cost,
                'signal_strength': signal.get('confidence', 0.5),
                'unrealized_pnl': 0.0,
                'status': 'open'
            }
            
            self.open_positions.append(position)
            self.current_balance -= transaction_cost  # Deduct transaction cost
            self.total_trades += 1
            
            logger.debug(f"Order executed: {direction} {position_units:.6f} units at {execution_price:.5f} "
                        f"(cost: ${transaction_cost:.2f})")
            
            return {
                'success': True,
                'position_id': position['id'],
                'execution_price': execution_price,
                'position_units': position_units,
                'transaction_cost': transaction_cost,
                'message': f"Order executed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f"Order execution failed"
            }
    
    def close_position(self, position_id: str, current_price: float, 
                      timestamp: pd.Timestamp, reason: str = "manual") -> Dict[str, Any]:
        """
        Close an open position
        
        Args:
            position_id: Position ID to close
            current_price: Current market price
            timestamp: Current timestamp
            reason: Reason for closing (stop_loss, take_profit, manual)
            
        Returns:
            Position close result
        """
        try:
            # Find the position
            position = None
            position_index = None
            for i, pos in enumerate(self.open_positions):
                if pos['id'] == position_id:
                    position = pos
                    position_index = i
                    break
            
            if not position:
                return {'success': False, 'error': 'Position not found'}
            
            # Apply slippage for closing
            slippage_factor = np.random.uniform(-self.slippage_pct, self.slippage_pct)
            if position['direction'] == 'BUY':
                exit_price = current_price * (1 - abs(slippage_factor))  # Sell at lower price
            else:
                exit_price = current_price * (1 + abs(slippage_factor))  # Cover at higher price
            
            # Calculate P&L
            if position['direction'] == 'BUY':
                pnl = (exit_price - position['entry_price']) * position['position_units']
            else:  # SELL
                pnl = (position['entry_price'] - exit_price) * position['position_units']
            
            # Calculate transaction cost for closing
            exit_transaction_cost = position['position_size_usd'] * self.transaction_cost_pct
            net_pnl = pnl - exit_transaction_cost
            
            # Update tracking
            self.total_fees += exit_transaction_cost
            self.current_balance += net_pnl
            
            if net_pnl > 0:
                self.winning_trades += 1
            
            # Create closed trade record
            closed_trade = {
                **position,
                'exit_price': exit_price,
                'exit_timestamp': timestamp,
                'exit_reason': reason,
                'exit_transaction_cost': exit_transaction_cost,
                'gross_pnl': pnl,
                'net_pnl': net_pnl,
                'total_transaction_costs': position['transaction_cost'] + exit_transaction_cost,
                'holding_period_minutes': (timestamp - position['entry_timestamp']).total_seconds() / 60,
                'status': 'closed'
            }
            
            self.closed_trades.append(closed_trade)
            self.open_positions.pop(position_index)
            
            logger.debug(f"Position closed: {position['direction']} {position['position_units']:.6f} units "
                        f"P&L: ${net_pnl:.2f} ({reason})")
            
            return {
                'success': True,
                'position_id': position_id,
                'exit_price': exit_price,
                'net_pnl': net_pnl,
                'gross_pnl': pnl,
                'exit_reason': reason
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'error': str(e)}
    
    def update_positions(self, current_price: float, timestamp: pd.Timestamp):
        """
        Update all open positions with current market data
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
        """
        total_unrealized_pnl = 0.0
        
        for position in self.open_positions:
            # Calculate unrealized P&L
            if position['direction'] == 'BUY':
                unrealized_pnl = (current_price - position['entry_price']) * position['position_units']
            else:  # SELL
                unrealized_pnl = (position['entry_price'] - current_price) * position['position_units']
            
            position['unrealized_pnl'] = unrealized_pnl
            total_unrealized_pnl += unrealized_pnl
        
        # Update current equity
        self.current_equity = self.current_balance + total_unrealized_pnl
        
        # Update drawdown tracking
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get current account information
        
        Returns:
            Account info dictionary
        """
        total_unrealized_pnl = sum([pos['unrealized_pnl'] for pos in self.open_positions])
        
        return {
            'initial_capital': self.initial_capital,
            'current_balance': self.current_balance,
            'current_equity': self.current_equity,
            'unrealized_pnl': total_unrealized_pnl,
            'total_return': (self.current_equity - self.initial_capital) / self.initial_capital,
            'total_return_pct': ((self.current_equity - self.initial_capital) / self.initial_capital) * 100,
            'open_positions_count': len(self.open_positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(1, len(self.closed_trades)),
            'total_fees': self.total_fees,
            'peak_equity': self.peak_equity,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown * 100
        }
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get list of open positions"""
        return self.open_positions.copy()
    
    def get_closed_trades(self) -> List[Dict[str, Any]]:
        """Get list of closed trades"""
        return self.closed_trades.copy()
    
    def close_all_positions(self, current_price: float, timestamp: pd.Timestamp) -> List[Dict[str, Any]]:
        """
        Close all open positions (useful for end of backtest)
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            List of close results
        """
        results = []
        open_position_ids = [pos['id'] for pos in self.open_positions.copy()]
        
        for position_id in open_position_ids:
            result = self.close_position(position_id, current_price, timestamp, reason="backtest_end")
            results.append(result)
        
        logger.info(f"Closed {len(results)} positions at backtest end")
        return results
    
    def reset(self):
        """Reset broker to initial state"""
        self.current_balance = self.initial_capital
        self.current_equity = self.initial_capital
        self.open_positions = []
        self.closed_trades = []
        self.trade_counter = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees = 0.0
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0
        
        logger.info("Simulation broker reset to initial state")