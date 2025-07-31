#!/usr/bin/env python3
"""
Simple P&L Test - Check if transactions costs are killing all profits
"""

def calculate_trade_pnl():
    """Calculate what a typical trade P&L looks like"""
    
    # Ultra config parameters
    entry_price = 1.2500
    position_size_usd = 800  # ~$800 position (8% of $10k)
    transaction_cost_pct = 0.0005  # 0.05%
    slippage_pct = 0.0002  # 0.02%
    
    # Original risk parameters that gave 0% win rate
    stop_loss_pct = 0.012  # 1.2%
    take_profit_pct = 0.035  # 3.5%
    
    print("💰 SIMPLE P&L ANALYSIS")
    print("=" * 50)
    print(f"Entry Price: {entry_price}")
    print(f"Position Size: ${position_size_usd}")
    print(f"Transaction Cost: {transaction_cost_pct*100}%")
    print(f"Slippage: {slippage_pct*100}%")
    print(f"Stop Loss: {stop_loss_pct*100}%")
    print(f"Take Profit: {take_profit_pct*100}%")
    print()
    
    # Calculate position units (for GBPUSD, this would be units of GBP)
    position_units = position_size_usd / entry_price
    
    # Entry costs
    entry_transaction_cost = position_size_usd * transaction_cost_pct
    entry_slippage = position_size_usd * slippage_pct
    total_entry_cost = entry_transaction_cost + entry_slippage
    
    print(f"📊 ENTRY COSTS:")
    print(f"  Transaction Cost: ${entry_transaction_cost:.2f}")
    print(f"  Slippage Cost: ${entry_slippage:.2f}")
    print(f"  Total Entry Cost: ${total_entry_cost:.2f}")
    print()
    
    # BUY scenario - Stop Loss Hit
    print("🔴 BUY TRADE - STOP LOSS HIT:")
    stop_loss_price = entry_price * (1 - stop_loss_pct)
    exit_slippage = position_size_usd * slippage_pct  # Sell at worse price
    exit_transaction_cost = position_size_usd * transaction_cost_pct
    
    # Calculate P&L
    gross_pnl = (stop_loss_price - entry_price) * position_units
    total_costs = entry_transaction_cost + exit_transaction_cost + entry_slippage + exit_slippage
    net_pnl = gross_pnl - total_costs
    
    print(f"  Stop Loss Price: {stop_loss_price:.5f}")
    print(f"  Gross P&L: ${gross_pnl:.2f}")
    print(f"  Total Costs: ${total_costs:.2f}")
    print(f"  Net P&L: ${net_pnl:.2f}")
    print(f"  Win: {'YES' if net_pnl > 0 else 'NO'}")
    print()
    
    # BUY scenario - Take Profit Hit
    print("🟢 BUY TRADE - TAKE PROFIT HIT:")
    take_profit_price = entry_price * (1 + take_profit_pct)
    
    # Calculate P&L
    gross_pnl = (take_profit_price - entry_price) * position_units
    net_pnl = gross_pnl - total_costs
    
    print(f"  Take Profit Price: {take_profit_price:.5f}")
    print(f"  Gross P&L: ${gross_pnl:.2f}")
    print(f"  Total Costs: ${total_costs:.2f}")
    print(f"  Net P&L: ${net_pnl:.2f}")
    print(f"  Win: {'YES' if net_pnl > 0 else 'NO'}")
    print()
    
    # Calculate break-even move required
    breakeven_move = total_costs / position_units
    breakeven_pct = breakeven_move / entry_price
    
    print(f"💡 BREAK-EVEN ANALYSIS:")
    print(f"  Break-even move required: {breakeven_move:.5f}")
    print(f"  Break-even percentage: {breakeven_pct*100:.3f}%")
    print(f"  Stop loss triggers at: {stop_loss_pct*100:.1f}%")
    print(f"  Take profit triggers at: {take_profit_pct*100:.1f}%")
    print()
    
    if stop_loss_pct < breakeven_pct:
        print("❌ PROBLEM: Stop loss triggers BEFORE break-even!")
        print(f"   Need minimum {breakeven_pct*100:.2f}% move to break even")
        print(f"   But stop loss hits at {stop_loss_pct*100:.1f}%")
    else:
        print("✅ Stop loss allows for break-even")
    
    return {
        'stop_loss_net_pnl': net_pnl,
        'breakeven_pct_required': breakeven_pct,
        'stop_loss_pct': stop_loss_pct,
        'costs_too_high': stop_loss_pct < breakeven_pct
    }

if __name__ == "__main__":
    results = calculate_trade_pnl()
    print(f"\n📋 Analysis: {results}")