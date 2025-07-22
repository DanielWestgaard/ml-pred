import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from broker.capital_com.capitalcom import CapitalCom
from utils.logging_utils import get_logger

logger = get_logger()

class OrderManager:
    """
    Manages order execution and communication with broker.
    Handles order placement, modification, and tracking.
    """
    
    def __init__(self, broker: str = 'capital_com', order_type: str = 'market',
                 execution_delay: float = 0.1, dry_run: bool = True, **kwargs):
        """
        Initialize order manager
        
        Args:
            broker: Broker name ('capital_com', etc.)
            order_type: Default order type ('market', 'limit')
            execution_delay: Delay between order operations in seconds
            dry_run: If True, simulate orders without actual execution
        """
        self.broker_name = broker
        self.order_type = order_type
        self.execution_delay = execution_delay
        self.dry_run = dry_run
        
        # Order tracking
        self.pending_orders = []
        self.executed_orders = []
        self.failed_orders = []
        
        # Initialize broker connection
        self.broker = self._initialize_broker()
        
        logger.info(f"Order manager initialized - Broker: {broker}, "
                    f"Order type: {order_type}, Dry run: {dry_run}")
    
    def _initialize_broker(self):
        """Initialize broker connection"""
        if self.broker_name == 'capital_com':
            if not self.dry_run:
                try:
                    broker = CapitalCom()
                    broker.start_session()  # Now automatically switches to USD_testing
                    logger.info("Capital.com broker connection established")
                    return broker
                except Exception as e:
                    logger.error(f"Failed to connect to Capital.com: {e}")
                    logger.info("Falling back to dry run mode")
                    self.dry_run = True
                    return None
            else:
                logger.info("Running in dry run mode - no actual orders will be placed")
                return None
        else:
            logger.warning(f"Unknown broker: {self.broker_name}, using dry run mode")
            self.dry_run = True
            return None
    
    def place_order(self, symbol: str, direction: str, size_usd: float, 
                   current_price: float, stop_loss: float = None, 
                   take_profit: float = None) -> Dict[str, Any]:
        """
        Place a trading order
        
        Args:
            symbol: Trading symbol (e.g., 'GBPUSD')
            direction: 'BUY' or 'SELL'
            size_usd: Position size in USD
            current_price: Current market price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order result information
        """
        order_id = f"{symbol}_{direction}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        order_info = {
            'order_id': order_id,
            'symbol': symbol,
            'direction': direction,
            'size_usd': size_usd,
            'order_type': self.order_type,
            'requested_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': datetime.now(),
            'status': 'pending'
        }
        
        self.pending_orders.append(order_info)
        
        try:
            if self.dry_run:
                # Simulate order execution
                result = self._simulate_order_execution(order_info)
            else:
                # Execute real order
                result = self._execute_real_order(order_info)
            
            # Update order status
            order_info.update(result)
            
            if result['success']:
                order_info['status'] = 'executed'
                self.executed_orders.append(order_info)
                logger.info(f"Order executed: {direction} {symbol} at {result.get('execution_price', current_price)}")
            else:
                order_info['status'] = 'failed'
                order_info['error_message'] = result.get('error', 'Unknown error')
                self.failed_orders.append(order_info)
                logger.error(f"Order failed: {order_info['error_message']}")
            
            # Remove from pending
            self.pending_orders.remove(order_info)
            
            # Execution delay
            time.sleep(self.execution_delay)
            
            return order_info
            
        except Exception as e:
            order_info['status'] = 'failed'
            order_info['error_message'] = str(e)
            self.failed_orders.append(order_info)
            self.pending_orders.remove(order_info)
            
            logger.error(f"Order execution failed: {e}")
            return order_info
    
    def _simulate_order_execution(self, order_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate order execution for dry run mode
        
        Args:
            order_info: Order information
            
        Returns:
            Simulated execution result
        """
        # Simulate small slippage
        slippage = 0.0002  # 2 pips
        execution_price = order_info['requested_price']
        
        if order_info['direction'] == 'BUY':
            execution_price += slippage
        else:
            execution_price -= slippage
        
        return {
            'success': True,
            'execution_price': execution_price,
            'execution_time': datetime.now(),
            'broker_order_id': f"SIM_{order_info['order_id']}",
            'slippage': abs(execution_price - order_info['requested_price']),
            'simulated': True
        }
    
    def _execute_real_order(self, order_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute real order through broker
        
        Args:
            order_info: Order information
            
        Returns:
            Execution result
        """
        if not self.broker:
            raise Exception("No broker connection available")
        
        try:
            # Convert size to broker format for Capital.com
            # For BTCUSD, Capital.com typically uses decimal sizes (e.g., 0.01 BTC minimum)
            if order_info['symbol'] == 'BTCUSD':
                # Calculate BTC units from USD amount
                btc_units = order_info['size_usd'] / order_info['requested_price']
                # Round to 4 decimal places and ensure minimum size of 0.01
                btc_units = max(0.01, round(btc_units, 4))
                size_units = str(btc_units)
                logger.info(f"BTCUSD size calculation: ${order_info['size_usd']} / ${order_info['requested_price']} = {btc_units} BTC")
            else:
                # For other symbols, use the original integer calculation
                size_units = str(max(1, int(order_info['size_usd'] / order_info['requested_price'])))
                logger.info(f"Size calculation for {order_info['symbol']}: ${order_info['size_usd']} / ${order_info['requested_price']} = {size_units} units")
            
            # Place market order
            if order_info['stop_loss'] and order_info['take_profit']:
                response = self.broker.place_market_order(
                    symbol=order_info['symbol'],
                    direction=order_info['direction'],
                    size=size_units,
                    stop_level=str(order_info['stop_loss']),
                    profit_level=str(order_info['take_profit'])
                )
            else:
                response = self.broker.place_market_order(
                    symbol=order_info['symbol'],
                    direction=order_info['direction'],
                    size=size_units
                )
            
            # Parse broker response (this would depend on actual broker API response format)
            if response and isinstance(response, dict) and 'dealReference' in response:
                return {
                    'success': True,
                    'execution_price': order_info['requested_price'],  # Would get from response
                    'execution_time': datetime.now(),
                    'broker_order_id': response.get('dealReference'),
                    'broker_response': response,
                    'simulated': False
                }
            else:
                error_msg = f"Invalid broker response. Type: {type(response)}, Content: {response}"
                return {
                    'success': False,
                    'error': error_msg,
                    'broker_response': response
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def close_position(self, symbol: str, position_id: str = None) -> Dict[str, Any]:
        """
        Close an existing position
        
        Args:
            symbol: Trading symbol
            position_id: Position identifier (optional)
            
        Returns:
            Close operation result
        """
        close_info = {
            'symbol': symbol,
            'position_id': position_id,
            'timestamp': datetime.now(),
            'status': 'pending'
        }
        
        try:
            if self.dry_run:
                # Simulate position close
                result = {
                    'success': True,
                    'close_price': 0.0,  # Would be current market price
                    'simulated': True
                }
            else:
                # Close real position
                if self.broker:
                    response = self.broker.close_all_orders(print_answer=False)
                    result = {
                        'success': True,
                        'broker_response': response,
                        'simulated': False
                    }
                else:
                    result = {
                        'success': False,
                        'error': 'No broker connection'
                    }
            
            close_info.update(result)
            
            if result['success']:
                close_info['status'] = 'closed'
                logger.info(f"Position closed for {symbol}")
            else:
                close_info['status'] = 'failed'
                logger.error(f"Failed to close position for {symbol}: {result.get('error', 'Unknown error')}")
            
            return close_info
            
        except Exception as e:
            close_info['status'] = 'failed'
            close_info['error_message'] = str(e)
            logger.error(f"Position close failed: {e}")
            return close_info
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get current account information from broker
        
        Returns:
            Account information
        """
        if self.dry_run:
            return {
                'balance': 10000.0,  # Simulated balance TODO: dynamic with broker
                'equity': 10000.0,
                'margin_used': 0.0,
                'margin_available': 10000.0,
                'simulated': True
            }
        
        try:
            if self.broker:
                account_info = self.broker.get_account_capital()
                return {
                    'balance': account_info.get('balance', 0),
                    'equity': account_info.get('equity', 0),
                    'margin_used': account_info.get('marginUsed', 0),
                    'margin_available': account_info.get('marginAvailable', 0),
                    'broker_response': account_info,
                    'simulated': False
                }
            else:
                return {'error': 'No broker connection'}
                
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {'error': str(e)}
    
    def get_order_summary(self) -> Dict[str, Any]:
        """
        Get summary of order execution statistics
        
        Returns:
            Order execution summary
        """
        total_orders = len(self.executed_orders) + len(self.failed_orders)
        success_rate = (len(self.executed_orders) / total_orders * 100) if total_orders > 0 else 0
        
        return {
            'total_orders': total_orders,
            'executed_orders': len(self.executed_orders),
            'failed_orders': len(self.failed_orders),
            'pending_orders': len(self.pending_orders),
            'success_rate_pct': success_rate,
            'dry_run_mode': self.dry_run
        }
    
    def cleanup(self, close_all_positions: bool = True):
        """
        Clean up broker connection and optionally close all positions
        
        Args:
            close_all_positions: Whether to close all open positions before cleanup (only in live mode)
        """
        if self.broker and not self.dry_run:
            try:
                # Close all positions before ending session (safety measure for live trading)
                if close_all_positions:
                    logger.info("🔄 Live trading cleanup: Closing all open positions...")
                    closed_success = self.broker.close_all_orders(print_answer=False)
                    if closed_success:
                        logger.info("✅ All positions successfully closed during cleanup")
                    else:
                        logger.warning("⚠️ Some positions may not have been closed during cleanup")
                
                # End broker session
                self.broker.end_session()
                logger.info("Broker session ended")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        elif not self.dry_run:
            logger.warning("No broker connection to clean up")
    
    def close_all_positions(self) -> Dict[str, Any]:
        """
        Emergency method to close all open positions (live trading only)
        
        Returns:
            Result of the close operation
        """
        if self.dry_run:
            logger.info("Dry run mode - no actual positions to close")
            return {'success': True, 'message': 'Dry run mode - simulated close all positions'}
        
        if not self.broker:
            return {'success': False, 'error': 'No broker connection available'}
        
        try:
            logger.info("🚨 EMERGENCY: Closing all open positions...")
            closed_success = self.broker.close_all_orders(print_answer=True)
            
            if closed_success:
                logger.info("✅ Emergency close: All positions successfully closed")
                return {'success': True, 'message': 'All positions closed successfully'}
            else:
                logger.error("❌ Emergency close: Failed to close some positions")
                return {'success': False, 'error': 'Failed to close some positions'}
        
        except Exception as e:
            logger.error(f"❌ Emergency close failed: {e}")
            return {'success': False, 'error': str(e)}