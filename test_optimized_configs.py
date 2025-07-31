#!/usr/bin/env python3
"""
Quick test to validate the optimized backtesting configurations.
"""

import subprocess
import sys
import os
from pathlib import Path
from utils.logging_utils import get_logger

logger = get_logger()

def run_backtest_config(config_name, config_path):
    """Run backtest with specified config and return results summary."""
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 TESTING: {config_name}")
    logger.info(f"Config: {config_path}")
    logger.info('='*60)
    
    try:
        # Run the main backtesting script
        result = subprocess.run([
            sys.executable, "main.py", 
            "--hyp", config_path
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        # Parse results from output
        output_lines = result.stdout.split('\n')
        
        trades = "Unknown"
        win_rate = "Unknown"
        return_pct = "Unknown"
        
        for line in output_lines:
            if "📊 Final Results:" in line:
                # Extract from format like: "📊 Final Results: Iterations: 3000/3000 | Return: -0.01%, Trades: 2, Win Rate: 0.0%"
                parts = line.split("|")
                for part in parts:
                    part = part.strip()
                    if "Trades:" in part:
                        trades = part.split(":")[-1].strip().replace(",", "")
                    elif "Win Rate:" in part:
                        win_rate = part.split(":")[-1].strip()
                    elif "Return:" in part:
                        return_pct = part.split(":")[-1].strip().replace(",", "")
        
        logger.info(f"✅ Results for {config_name}:")
        logger.info(f"   Trades: {trades}")
        logger.info(f"   Win Rate: {win_rate}")
        logger.info(f"   Return: {return_pct}")
        
        # Check if stderr has any critical errors
        if result.stderr and "ERROR" in result.stderr:
            logger.info(f"⚠️ Errors encountered: {result.stderr[:200]}...")
            
        return {
            'config': config_name,
            'trades': trades,
            'win_rate': win_rate,
            'return': return_pct,
            'success': result.returncode == 0
        }
        
    except subprocess.TimeoutExpired:
        logger.info(f"⏰ Test timed out for {config_name}")
        return {
            'config': config_name,
            'trades': 'TIMEOUT',
            'win_rate': 'TIMEOUT', 
            'return': 'TIMEOUT',
            'success': False
        }
    except Exception as e:
        logger.info(f"❌ Error testing {config_name}: {e}")
        return {
            'config': config_name,
            'trades': 'ERROR',
            'win_rate': 'ERROR',
            'return': 'ERROR', 
            'success': False
        }

def main():
    logger.info("🚀 Testing Optimized Backtesting Configurations")
    logger.info("This will validate that our fixes work correctly.\n")
    
    # Test configurations
    configs = [
        ("Original (Broken)", "config_backtest_v3_quick.yml"),
        ("Optimized (Fixed)", "config_backtest_v3_optimized.yml"),
        ("Basic (No Filtering)", "config_backtest_v3_basic.yml")
    ]
    
    results = []
    
    for config_name, config_file in configs:
        config_path = f"hypothesis/spread_compression_patterns/{config_file}"
        
        # Check if config exists
        if not os.path.exists(config_path):
            logger.info(f"❌ Config not found: {config_path}")
            continue
            
        result = run_backtest_config(config_name, config_path)
        results.append(result)
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 COMPARISON SUMMARY")
    logger.info('='*80)
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        logger.info(f"{status} {result['config']:<20} | Trades: {result['trades']:<8} | Win Rate: {result['win_rate']:<8} | Return: {result['return']}")
        
    logger.info(f"\n💡 EXPECTED RESULTS:")
    logger.info(f"   - Original: ~2-3 trades (temporal filtering issue)")
    logger.info(f"   - Optimized: ~10-50 trades (reduced filtering)")
    logger.info(f"   - Basic: ~100+ trades (no advanced filtering)")
    logger.info(f"\nIf you see these patterns, the fix is working! 🎉")

if __name__ == "__main__":
    main()