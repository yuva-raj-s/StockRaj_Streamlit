import yfinance as yf
import json
import os
import time
from typing import List, Dict
import threading
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PriceAlertManager:
    def __init__(self):
        self.alerts_file = "price_alerts.json"
        self.alerts = self._load_alerts()
        self._start_monitoring()
    
    def _load_alerts(self) -> List[Dict]:
        """Load alerts from file"""
        if os.path.exists(self.alerts_file):
            with open(self.alerts_file, 'r') as f:
                alerts = json.load(f)
                # Migrate existing alerts to include status field
                for alert in alerts:
                    if 'status' not in alert:
                        alert['status'] = 'active'
                    if 'created_at' not in alert:
                        alert['created_at'] = datetime.now().isoformat()
                    if 'triggered_at' not in alert:
                        alert['triggered_at'] = None
                    if 'triggered_price' not in alert:
                        alert['triggered_price'] = None
                return alerts
        return []
    
    def _save_alerts(self):
        """Save alerts to file"""
        with open(self.alerts_file, 'w') as f:
            json.dump(self.alerts, f)
    
    def add_alert(self, symbol: str, price: float, alert_type: str):
        """Add a new price alert"""
        # Ensure symbol has .NS suffix for NSE stocks
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        
        # Check if alert already exists
        for alert in self.alerts:
            if alert['symbol'] == symbol and alert['price'] == price and alert['type'] == alert_type:
                return False
        
        alert = {
            'symbol': symbol,
            'price': price,
            'type': alert_type,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'triggered_at': None,
            'triggered_price': None
        }
        self.alerts.append(alert)
        self._save_alerts()
        return True
    
    def remove_alert(self, symbol: str):
        """Remove an alert for a symbol"""
        # Ensure symbol has .NS suffix for NSE stocks
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        self.alerts = [alert for alert in self.alerts if alert['symbol'] != symbol]
        self._save_alerts()
    
    def get_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        try:
            return [alert for alert in self.alerts if alert.get('status') == 'active']
        except Exception as e:
            logger.error(f"Error getting alerts: {str(e)}")
            return []
    
    def _check_alerts(self):
        """Check if any alerts have been triggered"""
        if not self.alerts:
            return
        
        try:
            # Get all unique symbols from active alerts
            symbols = []
            for alert in self.alerts:
                if alert.get('status') == 'active':
                    symbol = alert['symbol']
                    if not symbol.endswith('.NS'):
                        symbol = f"{symbol}.NS"
                    symbols.append(symbol)
            
            if not symbols:
                return
                
            # Create a dictionary of tickers for easier lookup
            tickers = {}
            for symbol in symbols:
                try:
                    tickers[symbol] = yf.Ticker(symbol)
                except Exception as e:
                    logger.error(f"Error creating ticker for {symbol}: {e}")
                    continue
            
            for alert in self.alerts:
                if alert.get('status') != 'active':
                    continue
                    
                try:
                    symbol = alert['symbol']
                    if not symbol.endswith('.NS'):
                        symbol = f"{symbol}.NS"
                    
                    if symbol not in tickers:
                        continue
                        
                    ticker = tickers[symbol]
                    
                    # Get current price using history
                    hist = ticker.history(period="1d", interval="1m")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        
                        # Check if alert condition is met
                        triggered = False
                        if alert['type'] == 'Above' and current_price >= alert['price']:
                            triggered = True
                        elif alert['type'] == 'Below' and current_price <= alert['price']:
                            triggered = True
                            
                        if triggered:
                            self._trigger_alert(alert, current_price)
                except Exception as e:
                    logger.error(f"Error checking alert for {alert['symbol']}: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
    
    def _trigger_alert(self, alert: Dict, current_price: float):
        """Handle triggered alert"""
        try:
            symbol = alert['symbol'].replace('.NS', '')
            alert['status'] = 'triggered'
            alert['triggered_at'] = datetime.now().isoformat()
            alert['triggered_price'] = current_price
            
            # Log the alert
            logger.info(f"ALERT TRIGGERED: {symbol} is now {alert['type']} {alert['price']} at {current_price}")
            
            # Here you could add notification logic (email, push notification, etc.)
            # For now, we'll just save the alert status
            self._save_alerts()
            
        except Exception as e:
            logger.error(f"Error triggering alert for {alert['symbol']}: {str(e)}")
    
    def _start_monitoring(self):
        """Start the alert monitoring thread"""
        def monitor():
            while True:
                try:
                    self._check_alerts()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Error in alert monitoring: {e}")
                    time.sleep(60)  # Wait longer if there's an error
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start() 