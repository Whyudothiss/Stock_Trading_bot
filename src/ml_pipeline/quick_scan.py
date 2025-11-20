#!/usr/bin/env python3
"""
Quick Stock Scanner - Test a few stocks quickly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ml_pipeline.scan_stocks import StockScanner


def main():
    """Quick scan of a few stocks"""
    
    # Get project root (two levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    
    # Test tickers - add your own here
    tickers = [
        'RIVN', 'ONDS', 'BW', 'BBAI', 'GSAT', 'LMND', 'AXTI',
        'TSEM', 'REAL', 'BE', 'GRAL', 'MU', 'WDC', 'SNDK', 'CSIQ'
    ]
    
    print("\n🔍 Quick Scan Test\n")
    print(f"Testing {len(tickers)} stocks: {', '.join(tickers)}\n")
    
    # Initialize scanner
    scanner = StockScanner()
    
    # Scan with 50% threshold (more permissive for testing)
    results = scanner.scan_batch(tickers, min_confidence=0.50)
    
    # Print all results
    scanner.print_results(results, top_n=len(results))
    
    # Save to project root results/scanners
    output_path = project_root / "results" / "scanners" / "quick_scan_results.json"
    scanner.save_results(results, str(output_path))


if __name__ == "__main__":
    main()
