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

def get_relative_time(published_date: datetime) -> str:
    """
    Convert datetime to relative time string (e.g., '2 mins ago', '1 hour ago')
    """
    now = datetime.now(pytz.UTC)
    diff = now - published_date
    
    if diff.days > 0:
        if diff.days == 1:
            return "1 day ago"
        return f"{diff.days} days ago"
    
    seconds = diff.seconds
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} {'min' if minutes == 1 else 'mins'} ago"
    else:
        hours = seconds // 3600
        return f"{hours} {'hour' if hours == 1 else 'hours'} ago"

def fetch_financial_news() -> List[Dict]:
    """
    Fetch real-time financial news from multiple sources.
    Returns headline, source, link, and published date.
    """
    try:
        news_sources = {
            'Google Finance': 'https://news.google.com/rss/search?q=indian+stocks&hl=en-IN&gl=IN&ceid=IN:en',
            'Money Control': 'https://www.moneycontrol.com/rss/latestnews.xml',
            'Economic Times': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'Financial Express': 'https://www.financialexpress.com/feed/',
            'Business Standard': 'https://www.business-standard.com/rss/markets-101.rss'
        }
        
        all_news = []
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        
        for source, url in news_sources.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    # Parse the published date
                    try:
                        # Try to parse the date with timezone
                        published_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z')
                    except:
                        try:
                            # Try parsing without timezone
                            published_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S')
                            # Make it timezone-aware with IST
                            published_date = ist.localize(published_date)
                        except:
                            # If all parsing fails, use current time
                            published_date = current_time
                    
                    # Only add news from the last 24 hours
                    if (current_time - published_date).total_seconds() <= 86400:  # 24 hours in seconds
                        all_news.append({
                            'headline': entry.title,
                            'source': source,
                            'link': entry.link,
                            'published_date': published_date,
                            'relative_time': get_relative_time(published_date),
                            'description': entry.get('description', '')
                        })
            except Exception as e:
                logger.error(f"Error fetching news from {source}: {str(e)}")
                continue
        
        # Sort by published date (newest first)
        all_news.sort(key=lambda x: x['published_date'], reverse=True)
        return all_news
        
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        return []

# Create a cache for news with 5-minute TTL
news_cache = TTLCache(ttl_seconds=300)

def get_latest_news(limit: int = 5, offset: int = 0) -> List[Dict]:
    """
    Get latest financial news with pagination support.
    Uses cache to prevent too frequent API calls.
    """
    try:
        # Try to get from cache first
        cached_news = news_cache.get('latest_news')
        if cached_news is None:
            # If not in cache, fetch new data
            all_news = fetch_financial_news()
            news_cache.set('latest_news', all_news)
        else:
            all_news = cached_news
        
        # Apply pagination
        return all_news[offset:offset + limit]
        
    except Exception as e:
        logger.error(f"Error getting latest news: {str(e)}")
        return [] 