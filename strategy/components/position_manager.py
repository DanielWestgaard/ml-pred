import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from utils.logging_utils import get_logger

logger = get_logger()

class PositionManager:
    """
    Manages portfolio positions including tracking, updating, and reporting.
    TODO: Also integrate with the broker to make sure tracking is correct and state is as intended?
    """
    
    def __init__(self, position_tracking: str = 'detailed', 
                 max_concurrent_positions: int = 3, **kwargs):
        """
        Initialize position manager
        TODO: Not working with broker (capital.com) API yet.
        
        Args:
            position_tracking: Level of tracking ('basic', 'detailed')
            max_concurrent_positions: Maximum number of concurrent positions
        """
        self.position_tracking = position_tracking
        self.max_concurrent_positions = max_concurrent_positions
        
        # Position storage
        self.open_positions = []
        self.closed_positions = []
        self.position_history = []
        
        # Performance tracking
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        logger.info(f"Position manager initialized - Tracking: {position_tracking}, "
                    f"Max positions: {max_concurrent_positions}")
    
    def open_position(self, signal_info: Dict[str, Any], position_size_info: Dict[str, Any],
                     current_price: float, symbol: str) -> Dict[str, Any]:
        """
        Open a new position
        
        Args:
            signal_info: Information about the trading signal
            position_size_info: Position sizing information from risk manager
            current_price: Current market price
            symbol: Trading symbol
            
        Returns:
            Position information
        """
        position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        position = {
            'position_id': position_id,
            'symbol': symbol,
            'direction': signal_info['direction'],
            'entry_price': current_price,
            'position_size_usd': position_size_info['position_size_usd'],
            'position_units': position_size_info['position_units'],
            'stop_loss_price': position_size_info['stop_loss_price'],
            'take_profit_price': position_size_info['take_profit_price'],
            'entry_time': datetime.now(),
            'signal_strength': signal_info['strength'],
            'signal_confidence': signal_info['confidence'],
            'max_loss_usd': position_size_info['max_loss_usd'],
            'status': 'open',
            'current_pnl': 0.0,
            'current_price': current_price
        }
        
        # Add detailed tracking if enabled
        if self.position_tracking == 'detailed':
            position.update({
                'price_history': [current_price],
                'pnl_history': [0.0],
                'timestamp_history': [datetime.now()],
                'max_favorable_excursion': 0.0,
                'max_adverse_excursion': 0.0
            })
        
        self.open_positions.append(position)
        self.position_history.append(position.copy())
        
        logger.info(f"Opened {position['direction']} position for {symbol} at {current_price} "
                    f"(Size: ${position_size_info['position_size_usd']:.2f})")
        
        return position
    
    def update_position(self, position_id: str, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Update an existing position with current market price
        
        Args:
            position_id: Position identifier
            current_price: Current market price
            
        Returns:
            Updated position or None if not found
        """
        for position in self.open_positions:
            if position['position_id'] == position_id:
                # Update current price
                position['current_price'] = current_price
                
                # Calculate current P&L
                entry_price = position['entry_price']
                units = position['position_units']
                
                if position['direction'] == 'BUY':
                    pnl = (current_price - entry_price) * units
                else:  # SELL
                    pnl = (entry_price - current_price) * units
                
                position['current_pnl'] = pnl
                
                # Update detailed tracking if enabled
                if self.position_tracking == 'detailed':
                    position['price_history'].append(current_price)
                    position['pnl_history'].append(pnl)
                    position['timestamp_history'].append(datetime.now())
                    
                    # Update excursion tracking
                    if pnl > position['max_favorable_excursion']:
                        position['max_favorable_excursion'] = pnl
                    if pnl < position['max_adverse_excursion']:
                        position['max_adverse_excursion'] = pnl
                
                return position
        
        logger.warning(f"Position {position_id} not found for update")
        return None
    
    def close_position(self, position_id: str, close_price: float, 
                      close_reason: str = 'manual') -> Optional[Dict[str, Any]]:
        """
        Close an existing position
        
        Args:
            position_id: Position identifier
            close_price: Closing price
            close_reason: Reason for closing ('stop_loss', 'take_profit', 'manual', etc.)
            
        Returns:
            Closed position or None if not found
        """
        for i, position in enumerate(self.open_positions):
            if position['position_id'] == position_id:
                # Update final price and P&L
                position['close_price'] = close_price
                position['close_time'] = datetime.now()
                position['close_reason'] = close_reason
                position['status'] = 'closed'
                
                # Calculate final P&L
                entry_price = position['entry_price']
                units = position['position_units']
                
                if position['direction'] == 'BUY':
                    final_pnl = (close_price - entry_price) * units
                else:  # SELL
                    final_pnl = (entry_price - close_price) * units
                
                position['final_pnl'] = final_pnl
                position['current_pnl'] = final_pnl
                
                # Calculate holding time
                holding_time = position['close_time'] - position['entry_time']
                position['holding_time_minutes'] = holding_time.total_seconds() / 60
                
                # Update performance tracking
                self.total_pnl += final_pnl
                self.total_trades += 1
                
                if final_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Move to closed positions
                self.closed_positions.append(position)
                self.open_positions.pop(i)
                
                logger.info(f"Closed {position['direction']} position for {position['symbol']} "
                            f"at {close_price} (P&L: ${final_pnl:.2f}, Reason: {close_reason})")
                
                return position
        
        logger.warning(f"Position {position_id} not found for closing")
        return None
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        return self.open_positions.copy()
    
    def get_position_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all open positions for a specific symbol"""
        return [pos for pos in self.open_positions if pos['symbol'] == symbol]
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get current portfolio summary
        
        Returns:
            Portfolio summary statistics
        """
        # Calculate unrealized P&L
        unrealized_pnl = sum([pos['current_pnl'] for pos in self.open_positions])
        
        # Calculate realized P&L from closed positions
        realized_pnl = sum([pos['final_pnl'] for pos in self.closed_positions])
        
        # Win rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Average trade
        avg_trade = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'open_positions_count': len(self.open_positions),
            'closed_positions_count': len(self.closed_positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': win_rate,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'total_pnl': self.total_pnl,
            'average_trade': avg_trade,
            'current_exposure': sum([pos['position_size_usd'] for pos in self.open_positions])
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics
        
        Returns:
            Detailed performance statistics
        """
        if not self.closed_positions:
            return {'message': 'No closed positions for performance analysis'}
        
        # Calculate metrics from closed positions
        pnls = [pos['final_pnl'] for pos in self.closed_positions]
        winning_trades_pnl = [pnl for pnl in pnls if pnl > 0]
        losing_trades_pnl = [pnl for pnl in pnls if pnl <= 0]
        
        # Basic metrics
        avg_win = sum(winning_trades_pnl) / len(winning_trades_pnl) if winning_trades_pnl else 0
        avg_loss = sum(losing_trades_pnl) / len(losing_trades_pnl) if losing_trades_pnl else 0
        
        # Risk metrics
        profit_factor = abs(sum(winning_trades_pnl) / sum(losing_trades_pnl)) if losing_trades_pnl else float('inf')
        
        # Holding time analysis
        holding_times = [pos['holding_time_minutes'] for pos in self.closed_positions]
        avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
        
        return {
            'total_trades': len(self.closed_positions),
            'winning_trades': len(winning_trades_pnl),
            'losing_trades': len(losing_trades_pnl),
            'win_rate': len(winning_trades_pnl) / len(self.closed_positions) * 100,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': sum(pnls),
            'average_holding_time_minutes': avg_holding_time,
            'largest_win': max(pnls) if pnls else 0,
            'largest_loss': min(pnls) if pnls else 0
        }
    
    def export_trade_log(self) -> pd.DataFrame:
        """
        Export trade history as DataFrame
        
        Returns:
            DataFrame with trade history
        """
        if not self.closed_positions:
            return pd.DataFrame()
        
        # Prepare data for DataFrame
        trade_data = []
        for pos in self.closed_positions:
            trade_data.append({
                'position_id': pos['position_id'],
                'symbol': pos['symbol'],
                'direction': pos['direction'],
                'entry_time': pos['entry_time'],
                'close_time': pos['close_time'],
                'entry_price': pos['entry_price'],
                'close_price': pos['close_price'],
                'position_size_usd': pos['position_size_usd'],
                'final_pnl': pos['final_pnl'],
                'close_reason': pos['close_reason'],
                'holding_time_minutes': pos['holding_time_minutes'],
                'signal_strength': pos['signal_strength'],
                'signal_confidence': pos['signal_confidence']
            })
        
        return pd.DataFrame(trade_data)