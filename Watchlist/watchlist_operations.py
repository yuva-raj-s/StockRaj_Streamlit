import yfinance as yf
import pandas as pd
from typing import List, Dict
import json
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WatchlistManager:
    def __init__(self):
        self.watchlist_file = "watchlist.json"
        self.watchlist = self._load_watchlist()
        
    def _load_watchlist(self) -> List[str]:
        """Load the watchlist from file"""
        if os.path.exists(self.watchlist_file):
            with open(self.watchlist_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_watchlist(self):
        """Save the watchlist to file"""
        with open(self.watchlist_file, 'w') as f:
            json.dump(self.watchlist, f)
    
    def search_stocks(self, query: str) -> List[Dict]:
        """Search for stocks using yfinance"""
        try:
            # Add .NS suffix for NSE stocks if not present
            if not query.endswith('.NS'):
                query = f"{query}.NS"
            
            search = yf.Ticker(query)
            info = search.info
            if info:
                return [{
                    'symbol': info.get('symbol', ''),
                    'shortname': info.get('shortName', ''),
                    'exchange': info.get('exchange', ''),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': info.get('dividendYield', 0)
                }]
            return []
        except Exception as e:
            logger.error(f"Error searching stocks: {e}")
            return []
    
    def add_to_watchlist(self, symbol: str):
        """Add a symbol to the watchlist"""
        # Ensure symbol has .NS suffix for NSE stocks
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
            
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            self._save_watchlist()
    
    def remove_from_watchlist(self, symbol: str):
        """Remove a symbol from the watchlist"""
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            self._save_watchlist()
    
    def get_watchlist_data(self) -> pd.DataFrame:
        """Get current data for all stocks in watchlist"""
        if not self.watchlist:
            return pd.DataFrame()
        
        try:
            # Create a Tickers object for batch querying
            tickers = yf.Tickers(" ".join(self.watchlist))
            
            data = []
            for symbol in self.watchlist:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    
                    # Get the latest price data
                    hist = ticker.history(period="1d", interval="1m")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[0]
                        price_change = current_price - prev_close
                        price_change_pct = (price_change / prev_close * 100) if prev_close else 0
                        
                        # Get additional metrics
                        day_high = hist['High'].iloc[-1]
                        day_low = hist['Low'].iloc[-1]
                        
                        # Format market cap in crores
                        market_cap = info.get('marketCap', 0) / 10000000  # Convert to crores
                        
                        data.append({
                            'Symbol': symbol.replace('.NS', ''),
                            'Current Price': f"₹{current_price:.2f}",
                            'Change': f"₹{price_change:.2f}",
                            'Change %': f"{price_change_pct:.2f}%",
                            'Day High': f"₹{day_high:.2f}",
                            'Day Low': f"₹{day_low:.2f}",
                            'Market Cap': f"₹{market_cap:,.2f} Cr",
                            'P/E Ratio': f"{info.get('trailingPE', 0):.2f}",
                            'Dividend Yield': f"{info.get('dividendYield', 0):.2f}%"
                        })
                except Exception as e:
                    logger.error(f"Error getting data for {symbol}: {e}")
                    continue
            
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error getting watchlist data: {e}")
            return pd.DataFrame()
    
    def get_stock_analysis(self, symbol: str) -> Dict:
        """Get detailed analysis for a specific stock"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get historical data for technical analysis
            hist = ticker.history(period="1y")
            
            # Calculate technical indicators
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            
            # Get analyst recommendations
            recommendations = ticker.recommendations
            latest_recommendation = recommendations.iloc[-1] if not recommendations.empty else None
            
            # Get earnings dates
            earnings_dates = ticker.earnings_dates
            next_earnings = earnings_dates.iloc[0] if not earnings_dates.empty else None
            
            # Get sustainability data
            sustainability = ticker.sustainability
            
            return {
                'overview': {
                    'name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'eps': info.get('trailingEps', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'beta': info.get('beta', 0),
                    '52w_high': info.get('fiftyTwoWeekHigh', 0),
                    '52w_low': info.get('fiftyTwoWeekLow', 0)
                },
                'technical': {
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'sma_200': sma_200,
                    'trend': 'Bullish' if sma_20 > sma_50 > sma_200 else 'Bearish' if sma_20 < sma_50 < sma_200 else 'Neutral'
                },
                'recommendations': {
                    'current': latest_recommendation['To Grade'] if latest_recommendation is not None else 'N/A',
                    'target_price': latest_recommendation['Price Target'] if latest_recommendation is not None else None,
                    'date': latest_recommendation.name.strftime("%Y-%m-%d") if latest_recommendation is not None else None
                },
                'earnings': {
                    'next_date': next_earnings.name.strftime("%Y-%m-%d") if next_earnings is not None else None,
                    'estimate': next_earnings['EPS Estimate'] if next_earnings is not None else None
                },
                'sustainability': {
                    'environment_score': sustainability.get('environmentScore', 0) if sustainability is not None else None,
                    'social_score': sustainability.get('socialScore', 0) if sustainability is not None else None,
                    'governance_score': sustainability.get('governanceScore', 0) if sustainability is not None else None
                }
            }
        except Exception as e:
            logger.error(f"Error getting stock analysis for {symbol}: {e}")
            return None 