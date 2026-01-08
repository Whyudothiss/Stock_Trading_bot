// API base URL
const API_BASE = '';

// State
let refreshInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing dashboard...');
    
    // Set up event listeners
    setupEventListeners();
    
    // Initial data load
    refreshDashboard();
    
    // Auto-refresh every 5 seconds
    refreshInterval = setInterval(refreshDashboard, 5000);
});

function setupEventListeners() {
    // Scanner controls
    document.getElementById('start-scanner').addEventListener('click', startScanner);
    document.getElementById('stop-scanner').addEventListener('click', stopScanner);
    
    // Add ticker
    document.getElementById('add-ticker').addEventListener('click', addTicker);
    document.getElementById('ticker-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') addTicker();
    });
    
    // Hotlist controls
    document.getElementById('revalidate-hotlist').addEventListener('click', revalidateHotlist);
    document.getElementById('clear-hotlist').addEventListener('click', clearHotlist);
}

async function refreshDashboard() {
    try {
        await Promise.all([
            updateStatus(),
            updateHotlist(),
            updateSignals()
        ]);
    } catch (error) {
        console.error('Error refreshing dashboard:', error);
    }
}

async function updateStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        
        // Update market status
        const marketStatus = document.getElementById('market-status');
        if (data.market_open) {
            marketStatus.textContent = '🟢 Market Open';
            marketStatus.className = 'status-badge market-open';
        } else {
            marketStatus.textContent = '🔴 Market Closed';
            marketStatus.className = 'status-badge market-closed';
        }
        
        // Update scanner status
        const scannerStatus = document.getElementById('scanner-status');
        if (data.scanner_running) {
            scannerStatus.textContent = 'Scanner: RUNNING';
            scannerStatus.className = 'status-badge scanner-running';
        } else {
            scannerStatus.textContent = 'Scanner: OFF';
            scannerStatus.className = 'status-badge';
        }
        
        // Update time
        const timeDisplay = document.getElementById('current-time');
        const time = new Date(data.current_time_et + ' EST').toLocaleTimeString('en-US', {
            timeZone: 'America/New_York',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        timeDisplay.textContent = time + ' ET';
        
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

async function updateHotlist() {
    try {
        const response = await fetch(`${API_BASE}/api/hotlist`);
        const data = await response.json();
        
        const hotlistDiv = document.getElementById('hotlist');
        const countSpan = document.getElementById('hotlist-count');
        
        countSpan.textContent = data.count;
        
        if (data.count === 0) {
            hotlistDiv.innerHTML = '<div class="loading">Hotlist is empty</div>';
            return;
        }
        
        hotlistDiv.innerHTML = data.hotlist.map(stock => `
            <div class="hotlist-item">
                <div class="hotlist-item-info">
                    <div class="hotlist-ticker">${stock.ticker}</div>
                    <div class="hotlist-prob">${(stock.probability * 100).toFixed(1)}% probability</div>
                    <div class="hotlist-meta">
                        Added: ${stock.added_date} | Range: ${stock.consol_range_pct?.toFixed(2)}%
                    </div>
                </div>
                <button class="hotlist-remove" onclick="removeTicker('${stock.ticker}')">
                    Remove
                </button>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Error updating hotlist:', error);
    }
}

async function updateSignals() {
    try {
        const response = await fetch(`${API_BASE}/api/signals`);
        const data = await response.json();
        
        const signalsDiv = document.getElementById('signals');
        const countSpan = document.getElementById('signals-count');
        
        countSpan.textContent = data.count;
        
        if (data.count === 0) {
            signalsDiv.innerHTML = '<div class="loading">No signals yet</div>';
            return;
        }
        
        signalsDiv.innerHTML = data.signals.map(signal => `
            <div class="signal-item">
                <div class="signal-header">
                    <span class="signal-ticker">${signal.ticker}</span>
                    <span class="signal-prob">${(signal.probability * 100).toFixed(1)}%</span>
                </div>
                <div class="signal-time">🕐 ${signal.timestamp}</div>
                <div class="signal-prices">
                    <div class="signal-price entry">
                        <div class="signal-price-label">Entry</div>
                        <div class="signal-price-value">$${signal.entry_price.toFixed(2)}</div>
                    </div>
                    <div class="signal-price stop">
                        <div class="signal-price-label">Stop</div>
                        <div class="signal-price-value">$${signal.stop_price.toFixed(2)}</div>
                    </div>
                    <div class="signal-price target">
                        <div class="signal-price-label">Target (5R)</div>
                        <div class="signal-price-value">$${signal.target_price.toFixed(2)}</div>
                    </div>
                </div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Error updating signals:', error);
    }
}

async function startScanner() {
    const messageDiv = document.getElementById('scanner-message');
    
    try {
        const response = await fetch(`${API_BASE}/api/scanner/start`, {
            method: 'POST'
        });
        const data = await response.json();
        
        showMessage(messageDiv, data.message, data.success ? 'success' : 'error');
        
        if (data.success) {
            // Refresh immediately to show running status
            updateStatus();
        }
    } catch (error) {
        showMessage(messageDiv, 'Error starting scanner', 'error');
        console.error('Error starting scanner:', error);
    }
}

async function stopScanner() {
    const messageDiv = document.getElementById('scanner-message');
    
    try {
        const response = await fetch(`${API_BASE}/api/scanner/stop`, {
            method: 'POST'
        });
        const data = await response.json();
        
        showMessage(messageDiv, data.message, data.success ? 'success' : 'error');
        
        if (data.success) {
            updateStatus();
        }
    } catch (error) {
        showMessage(messageDiv, 'Error stopping scanner', 'error');
        console.error('Error stopping scanner:', error);
    }
}

async function addTicker() {
    const input = document.getElementById('ticker-input');
    const ticker = input.value.trim().toUpperCase();
    const messageDiv = document.getElementById('add-message');
    
    if (!ticker) {
        showMessage(messageDiv, 'Please enter a ticker', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/hotlist/add`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ ticker })
        });
        const data = await response.json();
        
        showMessage(messageDiv, data.message, data.success ? 'success' : 'error');
        
        if (data.success) {
            input.value = '';
            updateHotlist();
        }
    } catch (error) {
        showMessage(messageDiv, 'Error adding ticker', 'error');
        console.error('Error adding ticker:', error);
    }
}

async function removeTicker(ticker) {
    try {
        const response = await fetch(`${API_BASE}/api/hotlist/remove`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ ticker })
        });
        const data = await response.json();
        
        if (data.success) {
            updateHotlist();
        }
    } catch (error) {
        console.error('Error removing ticker:', error);
    }
}

async function revalidateHotlist() {
    const button = document.getElementById('revalidate-hotlist');
    button.disabled = true;
    button.textContent = '🔄 Checking...';
    
    try {
        const response = await fetch(`${API_BASE}/api/hotlist/revalidate`, {
            method: 'POST'
        });
        const data = await response.json();
        
        // Show temporary message
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message success show';
        messageDiv.textContent = data.message;
        button.parentNode.insertBefore(messageDiv, button.nextSibling);
        
        setTimeout(() => messageDiv.remove(), 3000);
        
        updateHotlist();
    } catch (error) {
        console.error('Error revalidating hotlist:', error);
    } finally {
        button.disabled = false;
        button.textContent = '🔄 Revalidate';
    }
}

async function clearHotlist() {
    if (!confirm('Are you sure you want to clear the entire hotlist?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/hotlist/clear`, {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.success) {
            updateHotlist();
        }
    } catch (error) {
        console.error('Error clearing hotlist:', error);
    }
}

function showMessage(element, message, type) {
    element.textContent = message;
    element.className = `message ${type} show`;
    
    setTimeout(() => {
        element.classList.remove('show');
    }, 5000);
}
