#!/usr/bin/env python3
"""
Live Scanner Web Application
Flask backend with REST API for controlling the live trading scanner
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sys
import threading
import time
from pathlib import Path
from datetime import datetime
import pytz

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_bot.live_scanner import LiveScanner, HotlistManager
import yaml

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Global scanner instance
scanner = None
scanner_thread = None
scanner_running = False
scanner_lock = threading.Lock()

def load_config():
    """Load trading bot configuration"""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'trading_bot.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_scanner_background():
    """Run the scanner in live mode (background thread)"""
    global scanner_running
    
    try:
        if scanner:
            # Run live scanning (will block until market closes or stopped)
            scanner.scan_intraday_live()
    except Exception as e:
        print(f"❌ Scanner error: {e}")
    finally:
        scanner_running = False

# ============================================================================
# WEB ROUTES (HTML Pages)
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

# ============================================================================
# API ENDPOINTS (JSON)
# ============================================================================

@app.route('/api/status')
def api_status():
    """Get scanner status"""
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    
    return jsonify({
        'scanner_running': scanner_running,
        'market_open': scanner.is_market_open() if scanner else False,
        'current_time_et': now_et.strftime('%Y-%m-%d %H:%M:%S'),
        'hotlist_count': len(scanner.hotlist_manager.get_hotlist()) if scanner else 0,
        'signals_today': len(scanner.signals_today) if scanner else 0
    })

@app.route('/api/hotlist')
def api_hotlist():
    """Get current hotlist"""
    if not scanner:
        return jsonify({'error': 'Scanner not initialized'}), 500
    
    hotlist = scanner.hotlist_manager.get_hotlist()
    return jsonify({
        'hotlist': hotlist,
        'count': len(hotlist)
    })

@app.route('/api/hotlist/add', methods=['POST'])
def api_add_ticker():
    """Add ticker to hotlist"""
    if not scanner:
        return jsonify({'error': 'Scanner not initialized'}), 500
    
    data = request.json
    ticker = data.get('ticker', '').upper().strip()
    
    if not ticker:
        return jsonify({'error': 'Ticker is required'}), 400
    
    # Check Pipeline 1
    check_date = datetime.now()
    success, prob = scanner.hotlist_manager.add_ticker(ticker, check_date)
    
    if success:
        return jsonify({
            'success': True,
            'message': f'{ticker} added to hotlist',
            'probability': prob
        })
    elif prob is not None:
        threshold = scanner.config['pipeline1_threshold']
        return jsonify({
            'success': False,
            'message': f'{ticker} failed Pipeline 1 ({prob*100:.1f}% < {threshold*100:.0f}%)',
            'probability': prob
        })
    else:
        return jsonify({
            'success': False,
            'message': f'Failed to extract features for {ticker}'
        })

@app.route('/api/hotlist/remove', methods=['POST'])
def api_remove_ticker():
    """Remove ticker from hotlist"""
    if not scanner:
        return jsonify({'error': 'Scanner not initialized'}), 500
    
    data = request.json
    ticker = data.get('ticker', '').upper().strip()
    
    if not ticker:
        return jsonify({'error': 'Ticker is required'}), 400
    
    if scanner.hotlist_manager.remove_ticker(ticker):
        return jsonify({
            'success': True,
            'message': f'{ticker} removed from hotlist'
        })
    else:
        return jsonify({
            'success': False,
            'message': f'{ticker} not in hotlist'
        })

@app.route('/api/hotlist/revalidate', methods=['POST'])
def api_revalidate():
    """Revalidate all hotlist tickers"""
    if not scanner:
        return jsonify({'error': 'Scanner not initialized'}), 500
    
    check_date = datetime.now()
    removed = scanner.hotlist_manager.revalidate_all(check_date)
    
    return jsonify({
        'success': True,
        'removed': removed,
        'message': f'Removed {len(removed)} ticker(s)' if removed else 'All tickers still valid'
    })

@app.route('/api/hotlist/clear', methods=['POST'])
def api_clear_hotlist():
    """Clear entire hotlist"""
    if not scanner:
        return jsonify({'error': 'Scanner not initialized'}), 500
    
    scanner.hotlist_manager.clear_hotlist()
    return jsonify({
        'success': True,
        'message': 'Hotlist cleared'
    })

@app.route('/api/signals')
def api_signals():
    """Get signals generated today"""
    if not scanner:
        return jsonify({'error': 'Scanner not initialized'}), 500
    
    return jsonify({
        'signals': scanner.signals_today,
        'count': len(scanner.signals_today)
    })

@app.route('/api/scanner/start', methods=['POST'])
def api_start_scanner():
    """Start the live scanner"""
    global scanner_running, scanner_thread
    
    if not scanner:
        return jsonify({'error': 'Scanner not initialized'}), 500
    
    with scanner_lock:
        if scanner_running:
            return jsonify({
                'success': False,
                'message': 'Scanner is already running'
            })
        
        if not scanner.is_market_open():
            et_tz = pytz.timezone('US/Eastern')
            now_et = datetime.now(et_tz)
            return jsonify({
                'success': False,
                'message': f'Market is closed (current time: {now_et.strftime("%H:%M")} ET)'
            })
        
        # Start scanner in background thread
        scanner_running = True
        scanner_thread = threading.Thread(target=run_scanner_background, daemon=True)
        scanner_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Scanner started'
        })

@app.route('/api/scanner/stop', methods=['POST'])
def api_stop_scanner():
    """Stop the live scanner"""
    global scanner_running
    
    if not scanner:
        return jsonify({'error': 'Scanner not initialized'}), 500
    
    with scanner_lock:
        if not scanner_running:
            return jsonify({
                'success': False,
                'message': 'Scanner is not running'
            })
        
        scanner_running = False
        return jsonify({
            'success': True,
            'message': 'Scanner stopped'
        })

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_scanner():
    """Initialize the scanner on startup"""
    global scanner
    
    print("Initializing scanner...")
    config = load_config()
    scanner = LiveScanner(config, simulation_date=None, ws_mode=False)
    print("✅ Scanner initialized")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Initialize scanner
    initialize_scanner()
    
    # Run Flask app
    port = 5000
    print(f"\n🚀 Starting web server on http://localhost:{port}")
    print(f"📊 Dashboard: http://localhost:{port}/")
    print(f"\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
