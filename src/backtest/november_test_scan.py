#!/usr/bin/env python3
"""
October 2025 Test Scan - Run both pipelines on recent trades
Tests the live-ready model on October tickers to see which trades would be taken
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import numpy as np
from datetime import datetime
import json

# Import feature extractors
from ml_pipeline.consolidation_features import ConsolidationFeatureExtractor
from ml_pipeline.breakout_features import BreakoutFeatureExtractor


class NovemberTestScanner:
    """Test both pipelines on November 2025 trades"""
    
    def __init__(self):
        # Load models
        print("Loading models...")
        pipeline1_data = joblib.load('models/consolidation_detector.pkl')
        pipeline2_model = joblib.load('models/5R_predictor_5min.pkl')
        
        self.pipeline1_model = pipeline1_data['model']
        self.pipeline1_features = pipeline1_data['feature_names']
        
        # Pipeline 2 model is saved directly, load features separately
        self.pipeline2_model = pipeline2_model
        with open('models/5R_predictor_5min_features.txt', 'r') as f:
            self.pipeline2_features = [line.strip() for line in f]
        
        # Feature extractors
        self.consol_extractor = ConsolidationFeatureExtractor()
        self.breakout_extractor = BreakoutFeatureExtractor(timeframe='5Min')
        
        # Thresholds
        self.pipeline1_threshold = 0.70  # 70% consolidation confidence
        self.pipeline2_threshold = 0.30  # 30% 5R probability
        
        print(f"✅ Models loaded")
        print(f"   Pipeline 1: {len(self.pipeline1_features)} features")
        print(f"   Pipeline 2: {len(self.pipeline2_features)} features")
    
    def check_pipeline1(self, ticker: str, entry_date: str) -> tuple:
        """Check if consolidation passes Pipeline 1"""
        try:
            # Extract consolidation features
            features_dict = self.consol_extractor.extract_features(ticker, entry_date)
            if not features_dict:
                return False, 0.0
            
            # Prepare features in correct order
            feature_values = [features_dict.get(f, 0) for f in self.pipeline1_features]
            X = np.array([feature_values])
            
            # Get prediction
            prob = self.pipeline1_model.predict_proba(X)[0][1]
            
            return prob >= self.pipeline1_threshold, prob
            
        except Exception as e:
            print(f"   ⚠️  Pipeline 1 error: {e}")
            return False, 0.0
    
    def check_pipeline2(self, ticker: str, entry_date: str, entry_price: float,
                       stop_price: float, target_price: float) -> tuple:
        """Check if trade passes Pipeline 2"""
        try:
            # Extract breakout features
            features_dict = self.breakout_extractor.extract_features(
                ticker, entry_date, entry_price, stop_price, target_price
            )
            if not features_dict:
                return False, 0.0
            
            # Prepare features in correct order
            feature_values = [features_dict.get(f, 0) for f in self.pipeline2_features]
            X = np.array([feature_values])
            
            # Get prediction
            prob = self.pipeline2_model.predict_proba(X)[0][1]
            
            return prob >= self.pipeline2_threshold, prob
            
        except Exception as e:
            print(f"   ⚠️  Pipeline 2 error: {e}")
            return False, 0.0
    
    def scan_november_trades(self, tickers: list):
        """Scan November trades for the given tickers"""
        
        # Load all November 2025 trades
        november_trades = []
        with open('data/processed/ml_5R_labels_filtered.jsonl', 'r') as f:
            for line in f:
                trade = json.loads(line)
                entry_date = trade['entry_date'][:10]
                
                # Filter for November 2025 and specified tickers
                if entry_date.startswith('2025-11') and trade['ticker'] in tickers:
                    november_trades.append(trade)
        
        if not november_trades:
            print("\n❌ No November 2025 trades found for specified tickers")
            return
        
        print(f"\n{'='*80}")
        print(f"NOVEMBER 2025 TEST SCAN")
        print(f"{'='*80}")
        print(f"Found {len(november_trades)} trades in November 2025")
        print(f"Testing with Pipeline 1: {self.pipeline1_threshold*100:.0f}% threshold")
        print(f"Testing with Pipeline 2: {self.pipeline2_threshold*100:.0f}% threshold")
        print(f"{'='*80}\n")
        
        trades_taken = []
        trades_rejected = []
        
        for i, trade in enumerate(november_trades, 1):
            ticker = trade['ticker']
            entry_date = trade['entry_date'][:10]
            entry_price = trade['entry_price']
            stop_price = trade['initial_stop_loss_price']
            target_price = trade['ml_target']
            actual_result = trade['ml_label']
            
            print(f"[{i}/{len(november_trades)}] {ticker} on {entry_date}")
            print(f"   Entry: ${entry_price:.2f} | Stop: ${stop_price:.2f} | Target: ${target_price:.2f}")
            print(f"   Actual result: {actual_result}")
            
            # Check Pipeline 1
            p1_pass, p1_prob = self.check_pipeline1(ticker, entry_date)
            print(f"   🔍 Pipeline 1: {'✅ PASS' if p1_pass else '❌ REJECT'} ({p1_prob*100:.1f}%)")
            
            if not p1_pass:
                print()
                trades_rejected.append({
                    'ticker': ticker,
                    'date': entry_date,
                    'reason': 'Pipeline 1',
                    'p1_prob': p1_prob,
                    'actual': actual_result
                })
                continue
            
            # Check Pipeline 2
            p2_pass, p2_prob = self.check_pipeline2(
                ticker, entry_date, entry_price, stop_price, target_price
            )
            print(f"   🔍 Pipeline 2: {'✅ PASS' if p2_pass else '❌ REJECT'} ({p2_prob*100:.1f}%)")
            
            if not p2_pass:
                print()
                trades_rejected.append({
                    'ticker': ticker,
                    'date': entry_date,
                    'reason': 'Pipeline 2',
                    'p1_prob': p1_prob,
                    'p2_prob': p2_prob,
                    'actual': actual_result
                })
                continue
            
            # Both passed!
            print(f"   ✅ TRADE WOULD BE TAKEN!")
            print()
            
            trades_taken.append({
                'ticker': ticker,
                'date': entry_date,
                'entry': entry_price,
                'stop': stop_price,
                'target': target_price,
                'p1_prob': p1_prob,
                'p2_prob': p2_prob,
                'actual': actual_result
            })
        
        # Summary
        print(f"\n{'='*80}")
        print(f"📊 SCAN RESULTS")
        print(f"{'='*80}")
        print(f"Total trades evaluated: {len(november_trades)}")
        print(f"Trades TAKEN: {len(trades_taken)} ({len(trades_taken)/len(november_trades)*100:.1f}%)")
        print(f"Trades REJECTED: {len(trades_rejected)} ({len(trades_rejected)/len(november_trades)*100:.1f}%)")
        
        if trades_taken:
            print(f"\n🎯 TRADES THAT WOULD BE TAKEN:")
            print("-" * 80)
            winners = [t for t in trades_taken if t['actual'] == 'HIT_5R']
            losers = [t for t in trades_taken if t['actual'] != 'HIT_5R']
            
            for trade in trades_taken:
                outcome = "🎯 HIT 5R" if trade['actual'] == 'HIT_5R' else "🛑 STOPPED"
                print(f"{trade['ticker']:6s} on {trade['date']} - ${trade['entry']:.2f}")
                print(f"       P1: {trade['p1_prob']*100:5.1f}% | P2: {trade['p2_prob']*100:5.1f}% → {outcome}")
            
            print(f"\nPerformance if you took these trades:")
            print(f"   Winners: {len(winners)}/{len(trades_taken)} ({len(winners)/len(trades_taken)*100:.1f}%)")
            print(f"   Losers: {len(losers)}/{len(trades_taken)} ({len(losers)/len(trades_taken)*100:.1f}%)")
        else:
            print("\n❌ No trades would be taken with current thresholds")
        
        # Rejection breakdown
        if trades_rejected:
            p1_rejects = [t for t in trades_rejected if t['reason'] == 'Pipeline 1']
            p2_rejects = [t for t in trades_rejected if t['reason'] == 'Pipeline 2']
            
            print(f"\n📋 REJECTION BREAKDOWN:")
            print(f"   Pipeline 1 rejected: {len(p1_rejects)}")
            print(f"   Pipeline 2 rejected: {len(p2_rejects)}")
            
            # Show big winners that were missed
            missed_winners = [t for t in trades_rejected if t['actual'] == 'HIT_5R']
            if missed_winners:
                print(f"\n   ⚠️  5R winners MISSED: {len(missed_winners)}")
                for t in missed_winners[:5]:  # Show first 5
                    print(f"      {t['ticker']} on {t['date']} (rejected by {t['reason']})")
        
        print(f"{'='*80}\n")


def main():
    # User-specified tickers for November
    user_tickers = ['LMND', 'CIFR', 'REAL', 'BE', 'NUAI', 'ONDS', 'GRAL']
    
    # All historical tickers (286 total)
    all_historical_tickers = ['SOUN','CVNA','SERV','QUBT','RKLB','QBTS','MSTR','ARM','INDO','GME','BBAI','PLTR','LUNR','ALAB','TMC','IREN','OPEN','SMR','EVGO','PONY','AFRM','GCT','ZIM','AAOI','EH','CORZ','RDDT','ASTS','TEM','APLD','OKLO','HSAI','ARQQ','RZLV','ONDS','JOBY','UPST','SMCI','BE','CAVA','HIMS','AMPX','RGTI','SEDG','SES','NBIS','CRWV','TLRY','LMND','IOT','MARA','PRCH','DWAC','LCID','LAC','CXAI','ACB','DJT','RCAT','IONQ','TSLL','UAMY','RR','CTM','PLUG','MP','GLXY','SYM','CLSK','LYFT','APP','CAR','NNE','WULF','HUT','PSNL','WGS','PTON','GEVO','WOOF','CLOV','DXYZ','ACHR','SATL','PL','NEGG','QS','LIFW','SHOT','PDD','POWL','OSCR','AMSC','SOFI','UROY','NXT','RBLX','ROOT','CRWD','AXTI','CGC','SPOT','RKT','TOST','POET','VST','KOSS','LUMN','AGX','NIO','COMM','AUR','LTBR','INOD','RDW','BABX','UMAC','BKSY','EOSE','MBOT','WRD','NVTS','ZENA','UAVS','INVZ','LAES','SRFM','CRDO','OUST','PGY','AEHR','ZEPP','UUUU','AMDL','RUN','DVLT','UEC','TWST','UPWK','TLS','JAKK','DUOL','SHIP','PLSE','BLNK','SSNT','TPIC','BITF','MIGI','TREE','DRCT','ANY','HMST','SOXL','REKR','RUM','RSI','WIX','ACVA','SQ','VIPS','AI','BMR','STRC','ORCL','GXAI','HEAR','PAYS','HUYA','CENX','SE','VRT','HYMC','LLY','AGBA','ASPN','QCOM','FSLR','CDE','STRL','PAAS','SEZL','CAMT','SG','ONFO','NVDA','PEGY','LIDR','SMTC','OPRA','AVGO','BNAI','AMC','SILK','ELF','HOOD','BTDR','NFLX','GE','ZAPP','MELI','KVYO','PBI','BLND','LTH','CHWY','GDC','BOOT','NEON','RH','SWIM','KINS','PPTA','SATS','GDS','RAIL','BEKE','COMP','MULN','FREY','GGLL','QMCO','AVGX','MVST','OPTX','SIDU','CRNC','WKEY','SKYT','WBA','USEG','FFIE','TUYA','LX','CYD','CELH','QSG','HWM','GRPN','NUTX','CEP','PTIR','AISP','AEVA','GRYP','GRAL','CWRV','GRRR','MODV','USAR','COIN','CONL','FFAI','PDYN','CRCL','BTCS','WKHS','ARAI','DPRO','SOFX','FUBO','VNET','DOMO','LEU','SLDP','ROBN','KTOS','NIU','BULL','SY','VSAT','RERE','ABAT','METC','HOVR','HIVE','BMNR','BZAI','VERI','SNDK','BTQ','GPRO','PBF','NUAI','W','GSIT']
    
    # Combine and deduplicate
    all_tickers = list(set(user_tickers + all_historical_tickers))
    all_tickers = [t.upper() for t in all_tickers]  # Ensure uppercase
    
    print(f"Scanning {len(all_tickers)} tickers for November 2025 trades...")
    
    # Run scanner
    scanner = NovemberTestScanner()
    scanner.scan_november_trades(all_tickers)


if __name__ == "__main__":
    main()
