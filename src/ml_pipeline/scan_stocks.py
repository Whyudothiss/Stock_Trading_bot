#!/usr/bin/env python3
"""
Stock Scanner - Batch process hundreds of stocks to find consolidation patterns
Uses trained ML model to score each stock's current setup
"""

import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import List, Dict, Tuple
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ml_pipeline.consolidation_features import ConsolidationFeatureExtractor


class StockScanner:
    """Scan multiple stocks and rank by consolidation probability"""
    
    def __init__(self, model_path: str = None):
        """Load trained model"""
        if model_path is None:
            # Default to project root / models / consolidation_detector.pkl
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / "models" / "consolidation_detector.pkl"
        
        print(f"📦 Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.model = saved_data['model']
        self.feature_names = saved_data['feature_names']
        self.extractor = ConsolidationFeatureExtractor(lookback_days=10)
        print(f"✅ Model loaded with {len(self.feature_names)} features\n")
    
    def scan_ticker(self, ticker: str, date: str = None) -> Tuple[float, Dict]:
        """
        Scan single ticker and return probability + features
        
        Returns:
            (probability, features_dict) or (None, None) if failed
        """
        if date is None:
            # Use yesterday (most recent complete trading day)
            date_obj = datetime.now() - timedelta(days=1)
        else:
            # Convert string date to datetime object
            date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Extract features (expects datetime object)
        features = self.extractor.extract_features(ticker, date_obj)
        if features is None:
            return None, None
        
        # Convert to array in correct order
        feature_array = [features[name] for name in self.feature_names]
        
        # Get probability (probability of class 1 = consolidation)
        proba = self.model.predict_proba([feature_array])[0][1]
        
        return proba, features
    
    def scan_batch(self, 
                   tickers: List[str], 
                   min_confidence: float = 0.7,
                   date: str = None) -> List[Dict]:
        """
        Scan multiple tickers and return ranked results
        
        Args:
            tickers: List of ticker symbols to scan
            min_confidence: Minimum probability threshold (0-1)
            date: Date to scan (YYYY-MM-DD), defaults to yesterday
            
        Returns:
            List of dicts with ticker, probability, and key features
        """
        results = []
        
        print(f"🔍 Scanning {len(tickers)} stocks...")
        print(f"   Confidence threshold: {min_confidence*100:.0f}%")
        print(f"   Date: {date or 'yesterday'}\n")
        
        for i, ticker in enumerate(tickers, 1):
            # Progress indicator
            if i % 50 == 0:
                print(f"   Progress: {i}/{len(tickers)} stocks scanned...")
            
            try:
                proba, features = self.scan_ticker(ticker, date)
                
                if proba is None:
                    continue  # Skip failed tickers
                
                # Only keep if above threshold
                if proba >= min_confidence:
                    results.append({
                        'ticker': ticker,
                        'confidence': round(proba * 100, 2),  # Convert to percentage
                        'probability': round(proba, 4),
                        'features': {
                            'price_range_pct': round(features['price_range_pct'], 2),
                            'position_in_range': round(features['position_in_range'], 2),
                            'volatility': round(features['volatility'], 2),
                            'dist_from_sma10_pct': round(features['dist_from_sma10_pct'], 2),
                            'dist_from_sma20_pct': round(features['dist_from_sma20_pct'], 2),
                            'dist_from_sma50_pct': round(features['dist_from_sma50_pct'], 2),
                            'volume_trend': round(features['volume_trend'], 4),
                            'consolidation_range_pct': round(features['consolidation_range_pct'], 2),
                            'days_hugging_10sma': features['days_hugging_10sma'],
                        }
                    })
                
                # Rate limiting - be nice to Yahoo Finance
                time.sleep(0.5)
                
            except Exception as e:
                print(f"   ⚠️  {ticker}: {str(e)}")
                continue
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        print(f"\n✅ Scan complete: {len(results)} stocks above {min_confidence*100:.0f}% confidence\n")
        
        return results
    
    def print_results(self, results: List[Dict], top_n: int = 20):
        """Pretty print scan results"""
        if not results:
            print("❌ No stocks found above confidence threshold")
            return
        
        print(f"{'='*80}")
        print(f"TOP {min(top_n, len(results))} CONSOLIDATION CANDIDATES")
        print(f"{'='*80}\n")
        
        for i, stock in enumerate(results[:top_n], 1):
            print(f"{i}. {stock['ticker']:6s} - {stock['confidence']:5.1f}% confidence")
            print(f"   Range: {stock['features']['price_range_pct']:5.1f}% | "
                  f"Position: {stock['features']['position_in_range']:4.2f} | "
                  f"Volatility: {stock['features']['volatility']:4.2f}%")
            print(f"   10SMA: {stock['features']['dist_from_sma10_pct']:+5.1f}% | "
                  f"20SMA: {stock['features']['dist_from_sma20_pct']:+5.1f}% | "
                  f"50SMA: {stock['features']['dist_from_sma50_pct']:+5.1f}%")
            print(f"   Volume Trend: {stock['features']['volume_trend']:+.4f} | "
                  f"Consolidation Range: {stock['features']['consolidation_range_pct']:.2f}% | "
                  f"Hugging 10SMA: {stock['features']['days_hugging_10sma']}/10 days")
            print()
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save results to JSON file"""
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {output_path}")


def get_sp500_tickers() -> List[str]:
    """Get S&P 500 ticker list (simplified version)"""
    # Top 100 liquid S&P 500 stocks for testing
    # In production, you'd fetch full list from Wikipedia or other source
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ',
        'V', 'XOM', 'WMT', 'JPM', 'LLY', 'MA', 'PG', 'AVGO', 'HD', 'CVX',
        'MRK', 'COST', 'ABBV', 'PEP', 'KO', 'ADBE', 'CRM', 'MCD', 'CSCO', 'TMO',
        'ACN', 'ABT', 'NFLX', 'WFC', 'LIN', 'DHR', 'NKE', 'CMCSA', 'VZ', 'AMD',
        'PM', 'TXN', 'DIS', 'INTC', 'NEE', 'UPS', 'ORCL', 'RTX', 'COP', 'BA',
        'IBM', 'QCOM', 'HON', 'AMGN', 'INTU', 'UNP', 'GE', 'AMAT', 'LOW', 'SPGI',
        'CAT', 'SBUX', 'ELV', 'GS', 'DE', 'BKNG', 'AXP', 'MDT', 'BLK', 'TJX',
        'PLD', 'GILD', 'SYK', 'ADP', 'MMC', 'ISRG', 'ADI', 'VRTX', 'CI', 'MDLZ',
        'AMT', 'CVS', 'ZTS', 'LRCX', 'C', 'MO', 'REGN', 'SO', 'TMUS', 'BDX',
        'NOC', 'CB', 'EOG', 'PGR', 'SCHW', 'SLB', 'CME', 'DUK', 'BSX', 'PYPL'
    ]


def get_nasdaq100_tickers() -> List[str]:
    """Get NASDAQ 100 ticker list (simplified)"""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST',
        'ASML', 'NFLX', 'AMD', 'PEP', 'ADBE', 'CSCO', 'TMUS', 'CMCSA', 'INTC', 'TXN',
        'QCOM', 'HON', 'INTU', 'AMGN', 'AMAT', 'BKNG', 'ISRG', 'ADP', 'VRTX', 'SBUX',
        'GILD', 'PANW', 'ADI', 'LRCX', 'REGN', 'MU', 'MDLZ', 'PYPL', 'MELI', 'KLAC',
        'SNPS', 'CDNS', 'CRWD', 'MAR', 'CSX', 'CTAS', 'ORLY', 'ABNB', 'FTNT', 'NXPI',
        'ADSK', 'PCAR', 'MRVL', 'WDAY', 'DASH', 'CHTR', 'AEP', 'CPRT', 'MNST', 'PAYX',
        'ROST', 'KDP', 'FAST', 'ODFL', 'AZN', 'EA', 'BKR', 'GEHC', 'EXC', 'CTSH',
        'LULU', 'KHC', 'DXCM', 'TEAM', 'CSGP', 'CCEP', 'IDXX', 'XEL', 'ON', 'FANG',
        'ANSS', 'ZS', 'TTWO', 'DDOG', 'CDW', 'BIIB', 'WBD', 'GFS', 'MDB', 'ILMN',
        'MRNA', 'ZM', 'ENPH', 'LCID', 'RIVN', 'SIRI', 'JD', 'PDD', 'BIDU', 'NTES'
    ]


def main():
    """Main scanner execution"""
    print("\n" + "="*80)
    print("🔍 STOCK CONSOLIDATION SCANNER")
    print("="*80 + "\n")
    
    # Get project root (two levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    
    # Initialize scanner
    scanner = StockScanner()
    
    # Choose which list to scan
    print("Select ticker list to scan:")
    print("1. S&P 500 Top 100")
    print("2. NASDAQ 100")
    print("3. Custom list (edit code to add your tickers)")
    
    choice = input("\nEnter choice (1-3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        tickers = get_sp500_tickers()
        list_name = "S&P 500 Top 100"
    elif choice == "2":
        tickers = get_nasdaq100_tickers()
        list_name = "NASDAQ 100"
    else:
        # Example custom list
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        list_name = "Custom"
    
    print(f"\n📊 Scanning: {list_name} ({len(tickers)} stocks)")
    
    # Set confidence threshold
    threshold_input = input("Enter minimum confidence threshold (0-100) [default: 70]: ").strip()
    try:
        threshold = float(threshold_input) / 100 if threshold_input else 0.70
        threshold = max(0.0, min(1.0, threshold))  # Clamp to 0-1
    except:
        threshold = 0.70
    
    # Run scan
    start_time = time.time()
    results = scanner.scan_batch(tickers, min_confidence=threshold)
    elapsed = time.time() - start_time
    
    # Display results
    scanner.print_results(results, top_n=20)
    
    # Save results to project root
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = project_root / "results" / "scanners" / f"scan_results_{timestamp}.json"
    scanner.save_results(results, str(output_path))
    
    print(f"\n⏱️  Scan completed in {elapsed:.1f} seconds")
    print(f"📈 Success rate: {len(results)}/{len(tickers)} stocks "
          f"({len(results)/len(tickers)*100:.1f}% above threshold)")


if __name__ == "__main__":
    main()
