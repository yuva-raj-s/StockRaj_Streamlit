from typing import List, Dict
import pandas as pd
from datetime import datetime, timedelta
import time
import yfinance as yf
import logging
import pytz
from requests.exceptions import RequestException
import feedparser
from bs4 import BeautifulSoup
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom cache implementation with TTL
class TTLCache:
    def __init__(self, ttl_seconds=30):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (value, time.time())

# Global cache instance
stock_cache = TTLCache(ttl_seconds=30)

def get_stock_data(ticker: str, max_retries=3, retry_delay=1):
    """Get stock data with retries and error handling"""
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            time.sleep(0.5)  # Rate limiting
            return stock
        except RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(retry_delay)
    return None

def get_marquee_data(symbols: List[str]) -> List[Dict]:
    """
    Fetch Symbol, Current Price, Change % from yfinance for marquee display.
    """
    try:
        # Get current time in IST
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        
        data = []
        for symbol in symbols:
            try:
                stock = get_stock_data(symbol)
                if not stock:
                    continue
                    
                info = stock.info
                current_price = info.get('regularMarketPrice', 0)
                previous_close = info.get('regularMarketPreviousClose', 0)
                change_percent = ((current_price - previous_close) / previous_close) * 100
                
                data.append({
                    'symbol': symbol,
                    'name': info.get('shortName', symbol),
                    'current_price': round(current_price, 2),
                    'change_percent': round(change_percent, 2)
                })
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return data
    except Exception as e:
        logger.error(f"Error in marquee data: {str(e)}")
        return []

def get_market_overview() -> Dict:
    """
    Fetch real-time data for Nifty 50, Sensex, and Nifty Bank.
    """
    try:
        indices = {
            "Nifty 50": "^NSEI",
            "Sensex": "^BSESN",
            "Nifty Bank": "^NSEBANK"
        }
        
        data = {}
        for name, symbol in indices.items():
            try:
                stock = get_stock_data(symbol)
                if not stock:
                    continue
                    
                info = stock.info
                current_price = info.get('regularMarketPrice', 0)
                previous_close = info.get('regularMarketPreviousClose', 0)
                change_percent = ((current_price - previous_close) / previous_close) * 100
                
                data[name] = {
                    'price': round(current_price, 2),
                    'change_percent': round(change_percent, 2)
                }
            except Exception as e:
                logger.error(f"Error processing {name}: {str(e)}")
                continue
        
        return data
    except Exception as e:
        logger.error(f"Error in market overview: {str(e)}")
        return {}

def get_top_indian_stocks(limit: int = 10) -> List[Dict]:
    """
    Return top Indian stocks sorted by volume or gainers.
    Includes: Symbol, Name, Price, Change %, Volume.
    """
    try:
        # List of top Indian stocks
        symbols = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HDFC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "BAJFINANCE.NS",
            "WIPRO.NS", "HINDUNILVR.NS", "ITC.NS", "ASIANPAINT.NS", "MARUTI.NS"
        ]
        
        stocks_data = []
        for symbol in symbols:
            try:
                stock = get_stock_data(symbol)
                if not stock:
                    continue
                    
                info = stock.info
                current_price = info.get('regularMarketPrice', 0)
                previous_close = info.get('regularMarketPreviousClose', 0)
                change_percent = ((current_price - previous_close) / previous_close) * 100
                
                stocks_data.append({
                    'symbol': symbol,
                    'name': info.get('shortName', symbol.replace('.NS', '')),
                    'price': round(current_price, 2),
                    'change_percent': round(change_percent, 2),
                    'volume': info.get('regularMarketVolume', 0)
                })
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Sort by absolute change percentage and volume
        sorted_stocks = sorted(
            stocks_data,
            key=lambda x: (abs(x['change_percent']), x['volume']),
            reverse=True
        )
        return sorted_stocks[:limit]
        
    except Exception as e:
        logger.error(f"Error in top stocks: {str(e)}")
        return []

def fetch_financial_news() -> List[Dict]:
    """
    Scrape RSS feeds from Indian financial news sources.
    Returns headline, source, link.
    """
    try:
        news_sources = {
            'Bloomberg Quint': 'https://www.bloombergquint.com/feeds/sitemap_index.xml',
            'Economic Times': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'MoneyControl': 'https://www.moneycontrol.com/rss/latestnews.xml',
            'Google News': 'https://news.google.com/rss/search?q=indian+stocks&hl=en-IN&gl=IN&ceid=IN:en'
        }
        
        all_news = []
        for source, url in news_sources.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:  # Get top 5 news from each source
                    all_news.append({
                        'headline': entry.title,
                        'source': source,
                        'link': entry.link,
                        'published_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
            except Exception as e:
                logger.error(f"Error fetching news from {source}: {str(e)}")
                continue
        
        # Sort by published date
        all_news.sort(key=lambda x: x['published_date'], reverse=True)
        return all_news[:20]  # Return top 20 news items
        
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        return [] 