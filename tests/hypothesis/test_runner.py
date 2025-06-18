#!/usr/bin/env python3
"""
Simple test runner for hypothesis system tests.

This script runs hypothesis-related tests using subprocess calls to avoid
import path issues with unittest discovery.

Run with:
    python tests/hypothesis/test_runner.py
"""

import subprocess
import sys
import os


def run_test_module(module_path, description):
    """Run a test module and return the result"""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print('='*60)
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'unittest', module_path, '-v'
        ], capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), '../..'))
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {module_path}: {e}")
        return False


def main():
    """Run all hypothesis tests"""
    print("Hypothesis System Test Runner")
    print("=" * 60)
    
    # Test modules to run
    test_modules = [
        ('tests.hypothesis.unit_tests.test_config', 'Unit Tests - Configuration Parsing'),
        ('tests.hypothesis.integration_tests.test_hypothesis_flow', 'Integration Tests - Hypothesis Flows'),
        ('tests.utils.test_hypothesis_utils', 'Utility Tests - Hypothesis Utils')
    ]
    
    results = []
    total_success = True
    
    for module, description in test_modules:
        success = run_test_module(module, description)
        results.append((description, success))
        if not success:
            total_success = False
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description:<40} {status}")
    
    print(f"\n{'='*60}")
    if total_success:
        print("All hypothesis tests PASSED! ✅")
        print("\nThe hypothesis configuration system is working correctly:")
        print("- Configuration parsing is robust")
        print("- Integration flows work as expected")
        print("- Error handling is appropriate")
        print("- Utility functions are reliable")
        sys.exit(0)
    else:
        print("Some tests FAILED! ❌")
        print("Please review the output above for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()