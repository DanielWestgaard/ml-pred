import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class PositionStatus:
    """Data class for tracking position status"""
    position_id: str
    symbol: str
    direction: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    last_update: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    session: Optional[str] = None
    signal_confidence: Optional[float] = None
    max_favorable_excursion: float = 0.0  # Best profit reached
    max_adverse_excursion: float = 0.0    # Worst loss reached


class PositionMonitor:
    """
    Advanced position monitoring system for real-time trade management.
    Handles position tracking, risk monitoring, and automated trade management.
    """
    
    def __init__(self, update_interval: float = 1.0, **kwargs):
        """
        Initialize position monitor
        
        Args:
            update_interval: Seconds between position updates
        """
        self.update_interval = update_interval
        
        # Position tracking
        self.active_positions: Dict[str, PositionStatus] = {}
        self.closed_positions: List[PositionStatus] = []
        
        # Monitoring state
        self._is_monitoring = False
        self._monitor_thread = None
        self._position_lock = threading.RLock()
        
        # Risk monitoring
        self.total_exposure = 0.0
        self.unrealized_pnl = 0.0
        self.daily_pnl = 0.0
        self.max_drawdown_today = 0.0
        
        # Callbacks for position events
        self._position_callbacks = []
        self._risk_callbacks = []
        
        logging.info(f"Position monitor initialized - Update interval: {update_interval}s")
    
    def add_position_callback(self, callback):
        """Add callback for position updates"""
        self._position_callbacks.append(callback)
    
    def add_risk_callback(self, callback):
        """Add callback for risk events"""
        self._risk_callbacks.append(callback)
    
    def start_monitoring(self) -> bool:
        """Start position monitoring thread"""
        if self._is_monitoring:
            logging.warning("Position monitoring already active")
            return True
        
        try:
            self._is_monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_worker, daemon=True)
            self._monitor_thread.start()
            
            logging.info("Position monitoring started")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start position monitoring: {e}")
            self._is_monitoring = False
            return False
    
    def stop_monitoring(self):
        """Stop position monitoring"""
        if not self._is_monitoring:
            return
        
        logging.info("Stopping position monitoring...")
        self._is_monitoring = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        logging.info("Position monitoring stopped")
    
    def add_position(self, position_id: str, symbol: str, direction: str,
                    size: float, entry_price: float, **kwargs) -> bool:
        """
        Add new position to monitoring
        
        Args:
            position_id: Unique position identifier
            symbol: Trading symbol
            direction: 'BUY' or 'SELL'
            size: Position size
            entry_price: Entry price
            **kwargs: Additional position metadata
            
        Returns:
            True if position added successfully
        """
        try:
            with self._position_lock:
                if position_id in self.active_positions:
                    logging.warning(f"Position {position_id} already exists")
                    return False
                
                position = PositionStatus(
                    position_id=position_id,
                    symbol=symbol,
                    direction=direction,
                    size=size,
                    entry_price=entry_price,
                    current_price=entry_price,  # Initialize with entry price
                    unrealized_pnl=0.0,
                    entry_time=datetime.now(),
                    last_update=datetime.now(),
                    stop_loss=kwargs.get('stop_loss'),
                    take_profit=kwargs.get('take_profit'),
                    session=kwargs.get('session'),
                    signal_confidence=kwargs.get('signal_confidence')
                )
                
                self.active_positions[position_id] = position
                self._update_portfolio_metrics()
                
                logging.info(f"Added position to monitoring: {position_id} - "
                           f"{direction} {size} {symbol} @ {entry_price}")
                
                # Notify callbacks
                self._notify_position_callbacks('position_opened', position)
                
                return True
                
        except Exception as e:
            logging.error(f"Error adding position {position_id}: {e}")
            return False
    
    def update_position_price(self, position_id: str, current_price: float) -> bool:
        """Update position with current market price"""
        try:
            with self._position_lock:
                if position_id not in self.active_positions:
                    return False
                
                position = self.active_positions[position_id]
                position.current_price = current_price
                position.last_update = datetime.now()
                
                # Calculate unrealized PnL
                if position.direction.upper() == 'BUY':
                    position.unrealized_pnl = (current_price - position.entry_price) * position.size
                else:  # SELL
                    position.unrealized_pnl = (position.entry_price - current_price) * position.size
                
                # Update max favorable/adverse excursion
                if position.unrealized_pnl > position.max_favorable_excursion:
                    position.max_favorable_excursion = position.unrealized_pnl
                
                if position.unrealized_pnl < position.max_adverse_excursion:
                    position.max_adverse_excursion = position.unrealized_pnl
                
                # Check for stop loss or take profit triggers
                should_close = self._check_exit_conditions(position)
                
                if should_close:
                    self._trigger_position_close(position, should_close['reason'])
                
                return True
                
        except Exception as e:
            logging.error(f"Error updating position {position_id}: {e}")
            return False
    
    def close_position(self, position_id: str, close_price: float, reason: str = "manual") -> bool:
        """Close position and move to closed positions"""
        try:
            with self._position_lock:
                if position_id not in self.active_positions:
                    logging.warning(f"Position {position_id} not found for closing")
                    return False
                
                position = self.active_positions[position_id]
                position.current_price = close_price
                position.last_update = datetime.now()
                
                # Calculate final PnL
                if position.direction.upper() == 'BUY':
                    position.unrealized_pnl = (close_price - position.entry_price) * position.size
                else:  # SELL
                    position.unrealized_pnl = (position.entry_price - close_price) * position.size
                
                # Move to closed positions
                self.closed_positions.append(position)
                del self.active_positions[position_id]
                
                self._update_portfolio_metrics()
                
                logging.info(f"Closed position {position_id} - Reason: {reason}, "
                           f"PnL: {position.unrealized_pnl:.4f}")
                
                # Notify callbacks
                self._notify_position_callbacks('position_closed', position)
                
                return True
                
        except Exception as e:
            logging.error(f"Error closing position {position_id}: {e}")
            return False
    
    def _monitor_worker(self):
        """Worker thread for continuous position monitoring"""
        while self._is_monitoring:
            try:
                with self._position_lock:
                    # Update portfolio metrics
                    self._update_portfolio_metrics()
                    
                    # Check for risk limit breaches
                    self._check_risk_limits()
                    
                    # Age-based position management
                    self._check_position_aging()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logging.error(f"Error in position monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _check_exit_conditions(self, position: PositionStatus) -> Optional[Dict[str, str]]:
        """Check if position should be closed based on stop loss, take profit, etc."""
        try:
            # Stop loss check
            if position.stop_loss is not None:
                if position.direction.upper() == 'BUY' and position.current_price <= position.stop_loss:
                    return {'reason': 'stop_loss', 'price': position.stop_loss}
                elif position.direction.upper() == 'SELL' and position.current_price >= position.stop_loss:
                    return {'reason': 'stop_loss', 'price': position.stop_loss}
            
            # Take profit check
            if position.take_profit is not None:
                if position.direction.upper() == 'BUY' and position.current_price >= position.take_profit:
                    return {'reason': 'take_profit', 'price': position.take_profit}
                elif position.direction.upper() == 'SELL' and position.current_price <= position.take_profit:
                    return {'reason': 'take_profit', 'price': position.take_profit}
            
            return None
            
        except Exception as e:
            logging.error(f"Error checking exit conditions for {position.position_id}: {e}")
            return None
    
    def _check_risk_limits(self):
        """Check portfolio-level risk limits"""
        try:
            # Check total exposure
            if abs(self.total_exposure) > 0.1:  # 10% max exposure
                self._notify_risk_callbacks('high_exposure', {
                    'exposure': self.total_exposure,
                    'limit': 0.1
                })
            
            # Check drawdown
            if self.max_drawdown_today > 0.05:  # 5% max daily drawdown
                self._notify_risk_callbacks('max_drawdown', {
                    'drawdown': self.max_drawdown_today,
                    'limit': 0.05
                })
            
            # Check number of active positions
            if len(self.active_positions) > 5:  # Max 5 concurrent positions
                self._notify_risk_callbacks('too_many_positions', {
                    'count': len(self.active_positions),
                    'limit': 5
                })
                
        except Exception as e:
            logging.error(f"Error checking risk limits: {e}")
    
    def _check_position_aging(self):
        """Check for positions that have been open too long"""
        try:
            max_position_age = timedelta(hours=24)  # 24 hour max
            current_time = datetime.now()
            
            for position_id, position in list(self.active_positions.items()):
                age = current_time - position.entry_time
                
                if age > max_position_age:
                    logging.warning(f"Position {position_id} aged out after {age}")
                    self._trigger_position_close(position, 'aged_out')
                    
        except Exception as e:
            logging.error(f"Error checking position aging: {e}")
    
    def _trigger_position_close(self, position: PositionStatus, reason: str):
        """Trigger position close (to be implemented by order manager)"""
        logging.info(f"Triggering close for position {position.position_id} - Reason: {reason}")
        
        # Notify callbacks that position needs closing
        self._notify_position_callbacks('close_requested', position, {'reason': reason})
    
    def _update_portfolio_metrics(self):
        """Update portfolio-level metrics"""
        try:
            self.total_exposure = sum(pos.size for pos in self.active_positions.values())
            self.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            
            # Calculate daily PnL (including closed positions from today)
            today = datetime.now().date()
            today_closed_pnl = sum(
                pos.unrealized_pnl for pos in self.closed_positions
                if pos.last_update.date() == today
            )
            self.daily_pnl = self.unrealized_pnl + today_closed_pnl
            
            # Update max drawdown
            if self.daily_pnl < -abs(self.max_drawdown_today):
                self.max_drawdown_today = abs(self.daily_pnl)
                
        except Exception as e:
            logging.error(f"Error updating portfolio metrics: {e}")
    
    def _notify_position_callbacks(self, event_type: str, position: PositionStatus, extra_data: Dict = None):
        """Notify position event callbacks"""
        for callback in self._position_callbacks:
            try:
                callback(event_type, asdict(position), extra_data or {})
            except Exception as e:
                logging.error(f"Error in position callback: {e}")
    
    def _notify_risk_callbacks(self, event_type: str, data: Dict):
        """Notify risk event callbacks"""
        for callback in self._risk_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logging.error(f"Error in risk callback: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        with self._position_lock:
            return {
                'active_positions': len(self.active_positions),
                'total_exposure': self.total_exposure,
                'unrealized_pnl': self.unrealized_pnl,
                'daily_pnl': self.daily_pnl,
                'max_drawdown_today': self.max_drawdown_today,
                'positions': {pid: asdict(pos) for pid, pos in self.active_positions.items()},
                'last_update': datetime.now()
            }
    
    def get_position_details(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific position"""
        with self._position_lock:
            if position_id in self.active_positions:
                return asdict(self.active_positions[position_id])
            return None