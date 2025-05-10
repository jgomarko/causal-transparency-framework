#!/usr/bin/env python
"""
Test runner for the Causal Transparency Framework.

Run all unit tests with:
    python run_tests.py

Run a specific test module with:
    python run_tests.py tests.test_causal_discovery
"""

import sys
import unittest

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test module
        test_module = sys.argv[1]
        suite = unittest.defaultTestLoader.loadTestsFromName(test_module)
    else:
        # Run all tests
        suite = unittest.defaultTestLoader.discover('tests')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with non-zero code if there were failures
    sys.exit(not result.wasSuccessful())