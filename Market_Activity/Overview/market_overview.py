import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketOverview:
    def __init__(self):
        self.indices = {
            "NIFTY 50": "^NSEI",
            "SENSEX": "^BSESN",
            "NIFTY BANK": "^NSEBANK",
            "NIFTY 100": "^NSEI",
            "NIFTY 250": "^NSEI",
            "NIFTY MIDCAP 250": "^NSEI",
            "NIFTY MIDCAP 100": "^NSEI",
            "NIFTY SMALLCAP 100": "^NSEI",
            "NIFTY NEXT 50": "^NSEI"
        }
        
    def get_market_pulse(self):
        """Get market sentiment and major index summary"""
        try:
            nse = yf.Ticker("^NSEI")
            data = nse.history(period="1d")
            info = nse.info
            
            return {
                "current_price": data["Close"].iloc[-1],
                "change": data["Close"].iloc[-1] - data["Open"].iloc[0],
                "change_percent": ((data["Close"].iloc[-1] - data["Open"].iloc[0]) / data["Open"].iloc[0]) * 100,
                "day_high": data["High"].iloc[-1],
                "day_low": data["Low"].iloc[-1],
                "volume": data["Volume"].iloc[-1],
                "previous_close": data["Open"].iloc[0]
            }
        except Exception as e:
            logger.error(f"Error getting market pulse: {e}")
            return None

    def get_broad_market_indices(self):
        """Get price and change % for all major indices"""
        indices_data = {}
        for name, symbol in self.indices.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                
                indices_data[name] = {
                    "price": data["Close"].iloc[-1],
                    "change": data["Close"].iloc[-1] - data["Open"].iloc[0],
                    "change_percent": ((data["Close"].iloc[-1] - data["Open"].iloc[0]) / data["Open"].iloc[0]) * 100
                }
            except Exception as e:
                logger.error(f"Error getting data for {name}: {e}")
        return indices_data

    def get_market_volatility(self):
        """Get VIX index data"""
        try:
            vix = yf.Ticker("^INDIAVIX")
            data = vix.history(period="1d")
            
            return {
                "current_vix": data["Close"].iloc[-1],
                "change": data["Close"].iloc[-1] - data["Open"].iloc[0],
                "change_percent": ((data["Close"].iloc[-1] - data["Open"].iloc[0]) / data["Open"].iloc[0]) * 100,
                "previous_close": data["Open"].iloc[0]
            }
        except Exception as e:
            logger.error(f"Error getting VIX data: {e}")
            return None

    def get_market_movers(self):
        """Get top gainers, losers, 52w high/low, and trending stocks"""
        try:
            # List of major Indian stocks
            symbols = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "HDFC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "BAJFINANCE.NS",
                "WIPRO.NS", "HINDUNILVR.NS", "ITC.NS", "ASIANPAINT.NS", "MARUTI.NS"
            ]
            
            stocks_data = []
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    info = ticker.info
                    
                    if not hist.empty:
                        current_price = hist["Close"].iloc[-1]
                        previous_close = hist["Open"].iloc[0]
                        change_percent = ((current_price - previous_close) / previous_close) * 100
                        
                        stocks_data.append({
                            "symbol": symbol,
                            "name": info.get("longName", symbol),
                            "price": current_price,
                            "change_percent": change_percent,
                            "volume": hist["Volume"].iloc[-1],
                            "52w_high": info.get("fiftyTwoWeekHigh", 0),
                            "52w_low": info.get("fiftyTwoWeekLow", 0)
                        })
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Sort stocks by different criteria
            sorted_by_change = sorted(stocks_data, key=lambda x: x["change_percent"], reverse=True)
            sorted_by_52w_high = sorted(stocks_data, key=lambda x: x["52w_high"], reverse=True)
            sorted_by_52w_low = sorted(stocks_data, key=lambda x: x["52w_low"])
            sorted_by_volume = sorted(stocks_data, key=lambda x: x["volume"], reverse=True)
            
            return {
                "top_gainers": sorted_by_change[:5],
                "top_losers": sorted_by_change[-5:],
                "52w_high": sorted_by_52w_high[:5],
                "52w_low": sorted_by_52w_low[:5],
                "trending": sorted_by_volume[:5]
            }
        except Exception as e:
            logger.error(f"Error getting market movers: {e}")
            return None

    def get_market_summary(self):
        """Get complete market overview"""
        return {
            "market_pulse": self.get_market_pulse(),
            "broad_indices": self.get_broad_market_indices(),
            "volatility": self.get_market_volatility(),
            "market_movers": self.get_market_movers()
        } 