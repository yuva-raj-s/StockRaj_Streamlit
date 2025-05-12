from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from textblob import TextBlob
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import plotly.graph_objects as go
from transformers import pipeline
from GoogleNews import GoogleNews
import torch
from AI_Chat.chatbot import IndianStockChatbot
from SentimentAnalysis.sentiment_analysis import analyze_asset_sentiment
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import logging
import requests
from urllib.parse import urlencode
import time
from functools import lru_cache
import random
import sys

# Get the absolute path of the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from sooraj.lstm_prediction import (
    calculate_technical_indicators,
    detect_anomalies,
    generate_signals,
    calculate_fibonacci_levels,
    prepare_data,
    create_lstm_model,
    predict_future_prices
)

app = FastAPI(
    title="StockRaj API",
    description="API for Indian Stock Market Analysis",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class StockInfo(BaseModel):
    symbol: str
    name: str
    price: float
    change_percent: float
    volume: Optional[int]
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    dividend_yield: Optional[float]

class MarketOverview(BaseModel):
    nifty: Dict[str, float]
    sensex: Dict[str, float]
    market_status: str
    advance_decline: Dict[str, Any]

class SentimentAnalysis(BaseModel):
    average_sentiment: float
    sentiment_label: str
    article_count: int

class TradingSignal(BaseModel):
    signal: str
    timestamp: datetime

# Initialize sentiment analysis model
SENTIMENT_ANALYSIS_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
device = 0 if torch.cuda.is_available() else -1
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_ANALYSIS_MODEL,
    device=device
)

# Initialize chatbot
chatbot = IndianStockChatbot()

# Technical Analysis Models
class StockRequest(BaseModel):
    ticker: str
    period: Optional[str] = "2y"

class ChatRequest(BaseModel):
    query: str

# Add this dictionary for common Indian stock symbols
COMMON_STOCK_SYMBOLS = {
    'RELIANCE': 'RELIANCE.NS',
    'TCS': 'TCS.NS',
    'INFY': 'INFY.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'ITC': 'ITC.NS',
    'SBIN': 'SBIN.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'WIPRO': 'WIPRO.NS',
    'AXISBANK': 'AXISBANK.NS',
    'ASIANPAINT': 'ASIANPAINT.NS',
    'MARUTI': 'MARUTI.NS',
    'HCLTECH': 'HCLTECH.NS',
    'SUNPHARMA': 'SUNPHARMA.NS',
    'TATASTEEL': 'TATASTEEL.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'TECHM': 'TECHM.NS',
    'TITAN': 'TITAN.NS'
}

# Add session management class
class YahooFinanceSession:
    def __init__(self):
        self.session = requests.Session()
        self.crumb = None
        self.cookie = None
        self.last_request_time = 0
        self.min_request_interval = 2  # Minimum seconds between requests
        
        # Common headers to mimic browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })

    def _get_crumb(self):
        """Get Yahoo Finance crumb for API requests"""
        try:
            url = "https://finance.yahoo.com/quote/AAPL"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Store cookies
            self.cookie = self.session.cookies.get_dict()
            
            # Extract crumb from response
            for line in response.text.split('\n'):
                if 'CrumbStore' in line:
                    self.crumb = line.split('"crumb":"')[1].split('"')[0]
                    break
            
            return self.crumb
        except Exception as e:
            logging.error(f"Error getting crumb: {str(e)}")
            return None

    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def get(self, url, params=None):
        """Make a GET request with rate limiting and crumb"""
        try:
            self._wait_for_rate_limit()
            
            # Add random delay to avoid pattern detection
            time.sleep(random.uniform(0.5, 1.5))
            
            # Ensure we have a crumb
            if not self.crumb:
                self._get_crumb()
            
            # Add crumb to params if we have one
            if self.crumb and params:
                params['crumb'] = self.crumb
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {str(e)}")
            # If we get a 401 or 403, try refreshing the crumb
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code in (401, 403):
                self.crumb = None
                self._get_crumb()
                return self.get(url, params)
            raise

# Create a session manager
class YahooFinanceManager:
    _instance = None
    _session = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YahooFinanceManager, cls).__new__(cls)
            cls._session = YahooFinanceSession()
        return cls._instance
    
    @classmethod
    def get_session(cls):
        if cls._session is None:
            cls._session = YahooFinanceSession()
        return cls._session

# Cache for stock data to reduce API calls
@lru_cache(maxsize=100)
def get_cached_stock_data(symbol: str, period: str) -> dict:
    """Cache stock data to reduce API calls"""
    session = YahooFinanceManager.get_session()
    
    base_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "range": period,
        "interval": "1d",
        "includePrePost": False,
        "events": "div,splits"
    }
    
    response = session.get(base_url, params=params)
    return response.json()

# Dashboard Endpoints
@app.get("/api/dashboard/marquee")
async def get_marquee_data():
    """Get live market data for marquee stocks"""
    try:
        symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
        data = []
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info and 'regularMarketPrice' in info:
                data.append({
                    'symbol': symbol,
                    'name': info.get('shortName', symbol),
                    'current_price': info['regularMarketPrice'],
                    'change_percent': info.get('regularMarketChangePercent', 0)
                })
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/market-overview")
async def get_market_overview():
    """Get market overview data for major indices"""
    try:
        indices = {
            'NIFTY 50': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTY BANK': '^NSEBANK'
        }
        
        data = {}
        for name, symbol in indices.items():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info and 'regularMarketPrice' in info:
                data[name] = {
                    'price': info['regularMarketPrice'],
                    'change_percent': info.get('regularMarketChangePercent', 0)
                }
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/top-stocks")
async def get_top_indian_stocks():
    """Get top performing Indian stocks"""
    try:
        symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS']
        data = []
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info and 'regularMarketPrice' in info:
                data.append({
                    'symbol': symbol,
                    'name': info.get('shortName', symbol),
                    'price': info['regularMarketPrice'],
                    'change_percent': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('regularMarketVolume', 0)
                })
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Market Activity Endpoints
@app.get("/api/market-activity/status")
async def get_market_status():
    """Get current market status"""
    try:
        nifty = yf.Ticker("^NSEI")
        info = nifty.info
        
        if 'regularMarketTime' in info:
            market_time = datetime.fromtimestamp(info['regularMarketTime'])
            now = datetime.now()
            
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            if market_open <= now <= market_close:
                return {"status": "Open"}
            else:
                return {"status": "Closed"}
        return {"status": "Unknown"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-activity/sector-performance")
async def get_sector_performance():
    """Get sector-wise performance"""
    try:
        sectors = {
            'NIFTY AUTO': 'NIFTYAUTO.NS',
            'NIFTY BANK': 'NIFTYBANK.NS',
            'NIFTY FMCG': 'NIFTYFMCG.NS',
            'NIFTY IT': 'NIFTYIT.NS',
            'NIFTY PHARMA': 'NIFTYPHARMA.NS',
            'NIFTY REALTY': 'NIFTYREALTY.NS'
        }
        
        data = []
        for name, symbol in sectors.items():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info and 'regularMarketPrice' in info:
                data.append({
                    'sector': name,
                    'price': info['regularMarketPrice'],
                    'change_percent': info.get('regularMarketChangePercent', 0)
                })
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-activity/market-pulse")
async def get_market_pulse():
    """Get market pulse data"""
    try:
        indices = {
            'NIFTY 50': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTY BANK': '^NSEBANK'
        }
        
        data = {}
        for name, symbol in indices.items():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info and 'regularMarketPrice' in info:
                data[name] = {
                    'price': info['regularMarketPrice'],
                    'change_percent': info.get('regularMarketChangePercent', 0)
                }
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-activity/broad-indices")
async def get_broad_indices():
    """Get broad market indices data"""
    try:
        indices = {
            'NIFTY 50': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTY BANK': '^NSEBANK',
            'NIFTY 100': '^NSEI100',
            'NIFTY 250': '^NSEI250',
            'NIFTY MIDCAP 250': '^NSEIMIDCAP250',
            'NIFTY MIDCAP 100': '^NSEIMIDCAP100',
            'NIFTY SMALLCAP 100': '^NSEISMALLCAP100',
            'NIFTY NEXT 50': '^NSEINEXT50'
        }
        
        data = {}
        for name, symbol in indices.items():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info and 'regularMarketPrice' in info:
                data[name] = {
                    'price': info['regularMarketPrice'],
                    'change_percent': info.get('regularMarketChangePercent', 0)
                }
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-activity/market-volatility")
async def get_market_volatility():
    """Get market volatility data"""
    try:
        vix = yf.Ticker("^VIX")
        info = vix.info
        if info and 'regularMarketPrice' in info:
            return {
                'vix': info['regularMarketPrice'],
                'change_percent': info.get('regularMarketChangePercent', 0)
            }
        return {"vix": 0, "change_percent": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-activity/market-movers")
async def get_market_movers():
    """Get market movers data"""
    try:
        symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS']
        data = []
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info and 'regularMarketPrice' in info:
                data.append({
                    'symbol': symbol,
                    'name': info.get('shortName', symbol),
                    'price': info['regularMarketPrice'],
                    'change_percent': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('regularMarketVolume', 0)
                })
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Watchlist Endpoints
@app.get("/api/watchlist")
async def get_watchlist():
    """Get user's watchlist"""
    try:
        if os.path.exists("Watchlist/watchlist.json"):
            with open("Watchlist/watchlist.json", "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/watchlist/add/{symbol}")
async def add_to_watchlist(symbol: str):
    """Add stock to watchlist"""
    try:
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or 'regularMarketPrice' not in info:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        watchlist = []
        if os.path.exists("Watchlist/watchlist.json"):
            with open("Watchlist/watchlist.json", "r") as f:
                watchlist = json.load(f)
        
        if symbol not in watchlist:
            watchlist.append(symbol)
            with open("Watchlist/watchlist.json", "w") as f:
                json.dump(watchlist, f)
        
        return {"message": f"{symbol} added to watchlist"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/watchlist/add-symbols")
async def add_symbols(symbols: List[str]):
    """Add symbols to the watchlist"""
    try:
        watchlist = []
        if os.path.exists("Watchlist/watchlist.json"):
            with open("Watchlist/watchlist.json", "r") as f:
                watchlist = json.load(f)
        
        for symbol in symbols:
            if symbol not in watchlist:
                watchlist.append(symbol)
        
        with open("Watchlist/watchlist.json", "w") as f:
            json.dump(watchlist, f)
        
        return {"message": f"{symbols} added to watchlist"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/watchlist/update")
async def update_watchlist():
    """Update the watchlist with current prices"""
    try:
        watchlist = []
        if os.path.exists("Watchlist/watchlist.json"):
            with open("Watchlist/watchlist.json", "r") as f:
                watchlist = json.load(f)
        
        data = []
        for symbol in watchlist:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info and 'regularMarketPrice' in info:
                data.append({
                    'symbol': symbol,
                    'name': info.get('shortName', symbol),
                    'price': info['regularMarketPrice'],
                    'change_percent': info.get('regularMarketChangePercent', 0)
                })
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Portfolio Endpoints
@app.get("/api/portfolio")
async def get_portfolio():
    """Get user's portfolio"""
    try:
        if os.path.exists("Portfolio/portfolio_data.json"):
            with open("Portfolio/portfolio_data.json", "r") as f:
                return json.load(f)
        return {"holdings": {}, "transactions": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/portfolio/add-transaction")
async def add_transaction(transaction: Dict[str, Any]):
    """Add a new transaction to the portfolio"""
    try:
        portfolio = {"holdings": {}, "transactions": []}
        if os.path.exists("Portfolio/portfolio_data.json"):
            with open("Portfolio/portfolio_data.json", "r") as f:
                portfolio = json.load(f)
        
        portfolio["transactions"].append(transaction)
        
        # Update holdings
        symbol = transaction["symbol"]
        quantity = transaction["quantity"]
        price = transaction["price"]
        
        if symbol in portfolio["holdings"]:
            portfolio["holdings"][symbol]["quantity"] += quantity
            portfolio["holdings"][symbol]["average_price"] = (
                (portfolio["holdings"][symbol]["average_price"] * 
                 (portfolio["holdings"][symbol]["quantity"] - quantity) +
                 price * quantity) / portfolio["holdings"][symbol]["quantity"]
            )
        else:
            portfolio["holdings"][symbol] = {
                "quantity": quantity,
                "average_price": price
            }
        
        with open("Portfolio/portfolio_data.json", "w") as f:
            json.dump(portfolio, f)
        
        return {"message": "Transaction added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/overview")
async def get_portfolio_overview():
    """Get portfolio overview data"""
    try:
        portfolio = {"holdings": {}, "transactions": []}
        if os.path.exists("Portfolio/portfolio_data.json"):
            with open("Portfolio/portfolio_data.json", "r") as f:
                portfolio = json.load(f)
        
        total_value = 0
        total_returns = 0
        for symbol, data in portfolio["holdings"].items():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info and 'regularMarketPrice' in info:
                current_price = info['regularMarketPrice']
                total_value += current_price * data["quantity"]
                total_returns += (current_price - data["average_price"]) * data["quantity"]
        
        return {
            "total_value": total_value,
            "total_returns": total_returns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/performance")
async def get_portfolio_performance():
    """Get portfolio performance data"""
    try:
        portfolio = {"holdings": {}, "transactions": []}
        if os.path.exists("Portfolio/portfolio_data.json"):
            with open("Portfolio/portfolio_data.json", "r") as f:
                portfolio = json.load(f)
        
        performance_data = []
        for symbol, data in portfolio["holdings"].items():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            if not hist.empty:
                performance_data.append({
                    'symbol': symbol,
                    'performance': hist.to_dict(orient='records')
                })
        return performance_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# AI Analysis Endpoints
@app.get("/api/ai-analysis/technical/{symbol}")
async def get_technical_analysis(symbol: str):
    """Get technical analysis for a stock"""
    try:
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo")
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Calculate technical indicators using ta library
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['Signal'] = macd.macd_signal()
        df['Histogram'] = macd.macd_diff()
        
        bb = BollingerBands(close=df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        
        return {
            "indicators": {
                "rsi": float(df['RSI'].iloc[-1]),
                "macd": float(df['MACD'].iloc[-1]),
                "signal": float(df['Signal'].iloc[-1]),
                "bb_upper": float(df['BB_Upper'].iloc[-1]),
                "bb_lower": float(df['BB_Lower'].iloc[-1]),
                "sma_20": float(df['SMA_20'].iloc[-1]),
                "ema_20": float(df['EMA_20'].iloc[-1])
            },
            "signals": get_trading_signals(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai-analysis/sentiment/{symbol}")
async def get_sentiment_analysis(symbol: str):
    """Get sentiment analysis for a stock"""
    try:
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            raise HTTPException(status_code=404, detail="No news available")
        
        sentiments = []
        for article in news[:10]:
            if 'title' in article:
                blob = TextBlob(article['title'])
                sentiments.append(blob.sentiment.polarity)
        
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            return {
                "average_sentiment": avg_sentiment,
                "sentiment_label": "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral",
                "article_count": len(sentiments)
            }
        raise HTTPException(status_code=404, detail="No sentiment data available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai-analysis/search-stocks")
async def search_stocks(query: str):
    """Search for stocks based on a query string"""
    try:
        symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS']
        results = []
        for symbol in symbols:
            if query.lower() in symbol.lower():
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if info and 'regularMarketPrice' in info:
                    results.append({
                        'symbol': symbol,
                        'name': info.get('shortName', symbol),
                        'price': info['regularMarketPrice'],
                        'change_percent': info.get('regularMarketChangePercent', 0)
                    })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai-analysis/technical-signals/{symbol}")
async def get_technical_signals(symbol: str):
    """Get technical signals for a stock"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo")
        if df.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Calculate technical indicators
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['Signal'] = macd.macd_signal()
        df['Histogram'] = macd.macd_diff()
        bb = BollingerBands(close=df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        
        return {
            "indicators": {
                "rsi": float(df['RSI'].iloc[-1]),
                "macd": float(df['MACD'].iloc[-1]),
                "signal": float(df['Signal'].iloc[-1]),
                "bb_upper": float(df['BB_Upper'].iloc[-1]),
                "bb_lower": float(df['BB_Lower'].iloc[-1])
            },
            "signals": get_trading_signals(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# AI Chat Endpoints
@app.post("/api/chat/query")
async def process_chat_query(query: str):
    """Process a chat query"""
    try:
        # Convert query to lowercase for easier matching
        query = query.lower()
        
        # Check for stock price query
        if "price" in query or "value" in query:
            words = query.split()
            for word in words:
                if word.upper() in ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]:
                    ticker = yf.Ticker(f"{word}.NS")
                    info = ticker.info
                    if info and 'regularMarketPrice' in info:
                        return {
                            "response": f"The current price of {info.get('shortName', word)} ({word}.NS) is ₹{info['regularMarketPrice']:,.2f} ({info.get('regularMarketChangePercent', 0):+.2f}%)"
                        }
        
        # Add more query processing logic here...
        
        return {"response": "I'm not sure about that. You can ask me about stock prices, market activity, sentiment analysis, your watchlist, or portfolio."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def get_trading_signals(df):
    """Generate trading signals based on technical indicators"""
    signals = []
    
    # RSI signals
    if df['RSI'].iloc[-1] > 70:
        signals.append("RSI indicates overbought conditions")
    elif df['RSI'].iloc[-1] < 30:
        signals.append("RSI indicates oversold conditions")
    
    # MACD signals
    if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] and df['MACD'].iloc[-2] <= df['Signal'].iloc[-2]:
        signals.append("MACD bullish crossover")
    elif df['MACD'].iloc[-1] < df['Signal'].iloc[-1] and df['MACD'].iloc[-2] >= df['Signal'].iloc[-2]:
        signals.append("MACD bearish crossover")
    
    # Bollinger Bands signals
    if df['Close'].iloc[-1] > df['BB_Upper'].iloc[-1]:
        signals.append("Price above upper Bollinger Band")
    elif df['Close'].iloc[-1] < df['BB_Lower'].iloc[-1]:
        signals.append("Price below lower Bollinger Band")
    
    return signals

# Add this new endpoint after the existing endpoints
@app.get("/api/market-activity/historical-indices")
async def get_historical_indices(period: str = "1day", interval: str = "15m"):
    """Get historical market indices data with specified period and interval"""
    try:
        indices = {
            'NIFTY 50': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTY BANK': '^NSEBANK'
        }
        
        data = {}
        for name, symbol in indices.items():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            if not hist.empty:
                data[name] = hist.to_dict(orient='records')
            else:
                data[name] = []
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add this new endpoint to return chart data
@app.get("/api/market-activity/chart")
async def get_chart_data(symbol: str, period: str = "1day", interval: str = "15m"):
    """Get chart data for a specific symbol with specified period and interval"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        return hist.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add this new endpoint for searching and looking up NSE stocks
@app.get("/api/stocks/search")
async def search_stocks(query: str):
    """Search for NSE stocks based on a query string"""
    try:
        # List of common NSE stock symbols
        symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS']
        
        results = []
        for symbol in symbols:
            if query.lower() in symbol.lower():
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if info and 'regularMarketPrice' in info:
                    results.append({
                        'symbol': symbol,
                        'name': info.get('shortName', symbol),
                        'price': info['regularMarketPrice'],
                        'change_percent': info.get('regularMarketChangePercent', 0)
                    })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add new endpoints for sentiment analysis
@app.get("/api/sentiment-analysis/{symbol}")
async def get_stock_sentiment(symbol: str):
    """Get sentiment analysis for a stock"""
    try:
        df, summary, sentiment_bar_img, sentiment_gauge, stock_chart, ticker = analyze_asset_sentiment(symbol)
        
        return {
            "status": "success",
            "symbol": symbol,
            "sentiment_summary": summary,
            "sentiment_data": df.to_dict('records'),
            "sentiment_bar": sentiment_bar_img,
            "sentiment_gauge": sentiment_gauge,
            "stock_chart": stock_chart,
            "ticker": ticker
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sentiment-analysis/news/{symbol}")
async def get_stock_news_sentiment(symbol: str):
    """Get news sentiment analysis for a stock"""
    try:
        googlenews = GoogleNews(lang='en', region='IN')
        googlenews.search(symbol)
        articles = googlenews.result()[:10]
        
        analyzed_articles = []
        for article in articles:
            sentiment = sentiment_analyzer(article["title"])[0]
            analyzed_articles.append({
                "title": article["title"],
                "description": article.get("desc", ""),
                "link": article["link"],
                "date": article["date"],
                "sentiment": sentiment["label"],
                "confidence": sentiment["score"]
            })
        
        return {
            "status": "success",
            "symbol": symbol,
            "articles": analyzed_articles
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add chatbot endpoint
@app.post("/api/chat")
async def chat_with_bot(request: ChatRequest):
    """Process chat queries"""
    try:
        response = chatbot.process_query(request.query)
        return {
            "status": "success",
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add technical analysis endpoints from Sooraj's code
@app.post("/api/technical-analysis/analyze-stock")
async def analyze_stock(request: StockRequest):
    """Analyze stock with technical indicators"""
    try:
        # Fetch and process data
        df = fetch_stock_data(request.ticker, request.period)
        df = calculate_indicators(df)
        fib_levels, trend, swing_high, swing_low, swing_high_date, swing_low_date = calculate_fibonacci_levels(df)
        extended_df, slope = calculate_future_projection(df, trend, swing_high, swing_low)
        df_with_signals = generate_signals(extended_df)
        
        # Convert dates back to string format for JSON serialization
        df_with_signals['Date'] = df_with_signals['Date'].apply(lambda x: mdates.num2date(x).strftime('%Y-%m-%d'))
        
        # Prepare the data for frontend
        chart_data = {
            "candlestick": df_with_signals[~df_with_signals['is_future']][['Date', 'Open', 'High', 'Low', 'Close']].to_dict('records'),
            "indicators": {
                "sma_20": df_with_signals[~df_with_signals['is_future']][['Date', 'SMA_20']].to_dict('records'),
                "ema_50": df_with_signals[~df_with_signals['is_future']][['Date', 'EMA_50']].to_dict('records'),
                "future_projection": df_with_signals[df_with_signals['is_future']][['Date', 'Projection_Line']].to_dict('records')
            },
            "signals": df_with_signals[~df_with_signals['is_future'] & (df_with_signals['Signal'] != "")][['Date', 'Signal', 'Close']].to_dict('records'),
            "fibonacci_levels": fib_levels,
            "analysis": {
                "trend": trend,
                "current_signal": df_with_signals[~df_with_signals['is_future'] & (df_with_signals['Signal'] != "")].tail(1)['Signal'].values[0] if not df_with_signals[~df_with_signals['is_future'] & (df_with_signals['Signal'] != "")].empty else "NEUTRAL",
                "current_price": float(df['Close'].iloc[-1]),
                "swing_high": float(swing_high),
                "swing_low": float(swing_low),
                "projection_slope": float(slope)
            }
        }
        
        return {
            "status": "success",
            "ticker": request.ticker,
            "data": chart_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/technical-analysis/indicators/{symbol}")
async def get_technical_indicators(symbol: str, period: str = "1mo"):
    """Get technical indicators for a stock"""
    try:
        df = fetch_stock_data(symbol, period)
        df = calculate_indicators(df)
        
        # Get the latest values and ensure they are JSON serializable
        latest_data = {
            "symbol": symbol,
            "sma_20": float(df['SMA_20'].iloc[-1]) if not pd.isna(df['SMA_20'].iloc[-1]) else 0,
            "ema_50": float(df['EMA_50'].iloc[-1]) if not pd.isna(df['EMA_50'].iloc[-1]) else 0,
            "macd": float(df['MACD'].iloc[-1]) if not pd.isna(df['MACD'].iloc[-1]) else 0,
            "macd_signal": float(df['MACD_Signal'].iloc[-1]) if not pd.isna(df['MACD_Signal'].iloc[-1]) else 0,
            "rsi": float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 0,
            "atr": float(df['ATR'].iloc[-1]) if not pd.isna(df['ATR'].iloc[-1]) else 0,
            "vwap": float(df['VWAP'].iloc[-1]) if not pd.isna(df['VWAP'].iloc[-1]) else 0,
            "vstop_long": float(df['VStop_Long'].iloc[-1]) if not pd.isna(df['VStop_Long'].iloc[-1]) else 0,
            "vstop_short": float(df['VStop_Short'].iloc[-1]) if not pd.isna(df['VStop_Short'].iloc[-1]) else 0
        }
        
        return {
            "status": "success",
            "data": latest_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/technical-analysis/signals/{symbol}")
async def get_trading_signals(symbol: str, period: str = "1mo"):
    """Get trading signals for a stock"""
    try:
        # Validate period
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        if period not in valid_periods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid period. Please use one of: {', '.join(valid_periods)}"
            )
        
        # Fetch and process data
        df = fetch_stock_data(symbol, period)
        df = calculate_indicators(df)
        df = generate_signals(df)
        
        # Get signals with additional information
        signals = []
        for idx, row in df[df['Signal'] != ""].iterrows():
            signal_info = {
                'date': mdates.num2date(row['Date']).strftime('%Y-%m-%d'),
                'signal': row['Signal'],
                'price': float(row['Close']),
                'strength': int(row['Signal_Strength']),
                'reason': row['Signal_Reason'],
                'indicators': {
                    'rsi': float(row['RSI']),
                    'macd': float(row['MACD']),
                    'macd_signal': float(row['MACD_Signal']),
                    'sma_20': float(row['SMA_20']),
                    'ema_50': float(row['EMA_50']),
                    'volume': float(row['Volume'])
                }
            }
            signals.append(signal_info)
        
        # Get current market status
        current_price = float(df['Close'].iloc[-1])
        current_sma = float(df['SMA_20'].iloc[-1])
        current_rsi = float(df['RSI'].iloc[-1])
        current_macd = float(df['MACD'].iloc[-1])
        current_macd_signal = float(df['MACD_Signal'].iloc[-1])
        
        market_status = {
            'current_price': current_price,
            'sma_20': current_sma,
            'price_vs_sma': f"{((current_price/current_sma - 1) * 100):.2f}%",
            'rsi': current_rsi,
            'macd_status': "Bullish" if current_macd > current_macd_signal else "Bearish",
            'trend': "Uptrend" if current_price > current_sma else "Downtrend"
        }
        
        # Get company info
        try:
            stock = yf.Ticker(validate_symbol(symbol))
            info = stock.info
            company_info = {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
        except:
            company_info = None
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "company_info": company_info,
            "signals": signals,
            "signal_count": len(signals),
            "last_signal": signals[-1] if signals else None,
            "period_analyzed": period,
            "market_status": market_status
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error in get_trading_signals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add helper functions for technical analysis
def validate_symbol(symbol: str) -> str:
    """Validate and format stock symbol"""
    try:
        # Convert to uppercase
        symbol = symbol.upper().strip()
        
        # If symbol already has .NS suffix, verify it exists
        if symbol.endswith('.NS'):
            base_symbol = symbol[:-3]
            if base_symbol in COMMON_STOCK_SYMBOLS:
                return symbol
        
        # Check if symbol exists in common symbols
        if symbol in COMMON_STOCK_SYMBOLS:
            return COMMON_STOCK_SYMBOLS[symbol]
        
        # If not in common symbols, try to verify with yfinance
        test_symbol = f"{symbol}.NS"
        stock = yf.Ticker(test_symbol)
        info = stock.info
        
        if info and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            return test_symbol
        
        raise ValueError(f"Invalid symbol: {symbol}")
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Stock symbol not found or invalid: {symbol}. Please use valid NSE symbols like RELIANCE, TCS, INFY, etc."
        )

def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance for Indian stocks"""
    try:
        # Validate and format the symbol
        valid_symbol = validate_symbol(ticker)
        
        try:
            # Try to get data from cache first
            data = get_cached_stock_data(valid_symbol, period)
            
            # Convert response to DataFrame
            chart = data['chart']['result'][0]
            timestamps = pd.to_datetime(chart['timestamp'], unit='s')
            
            df = pd.DataFrame({
                'Date': timestamps,
                'Open': chart['indicators']['quote'][0]['open'],
                'High': chart['indicators']['quote'][0]['high'],
                'Low': chart['indicators']['quote'][0]['low'],
                'Close': chart['indicators']['quote'][0]['close'],
                'Volume': chart['indicators']['quote'][0]['volume']
            })
            
        except Exception as e:
            logging.warning(f"Cache/API fetch failed, falling back to yfinance: {str(e)}")
            # Fallback to yfinance if direct API fails
            stock = yf.Ticker(valid_symbol)
            df = stock.history(period=period, interval="1d")
            df.reset_index(inplace=True)
        
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol: {ticker}. Please verify the symbol and try again."
            )
        
        # Convert dates to numeric format for technical analysis
        df['Date'] = df['Date'].map(mdates.date2num)
        
        return df
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error fetching stock data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching stock data: {str(e)}"
        )

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for daily timeframe"""
    try:
        # Ensure we have enough data
        if len(df) < 60:  # Minimum required for all our calculations
            raise HTTPException(
                status_code=400,
                detail="Not enough data points for technical analysis. Please use a longer time period."
            )

        # Moving Averages with safety checks
        df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        df['EMA_50'] = EMAIndicator(df['Close'], window=50).ema_indicator()
        
        # MACD
        macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # RSI
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        
        # Volatility Stop (ATR based)
        atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
        df['ATR'] = atr.average_true_range()
        
        # Safe calculations for Volatility Stop
        df['VStop_Long'] = df['High'].rolling(14).max().fillna(df['High']) - 3 * df['ATR'].fillna(0)
        df['VStop_Short'] = df['Low'].rolling(14).min().fillna(df['Low']) + 3 * df['ATR'].fillna(0)
        
        # Calculate VWAP safely
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VP'] = df['Typical_Price'] * df['Volume'].fillna(0)
        
        # Safe cumulative calculations
        df['Cumulative_VP'] = df['VP'].rolling(window=20, min_periods=1).sum()
        df['Cumulative_Volume'] = df['Volume'].rolling(window=20, min_periods=1).sum()
        
        # Safe VWAP calculation
        df['VWAP'] = np.where(
            df['Cumulative_Volume'] > 0,
            df['Cumulative_VP'] / df['Cumulative_Volume'],
            df['Typical_Price']
        )
        
        # Clean up intermediate columns
        df = df.drop(['Typical_Price', 'VP', 'Cumulative_VP', 'Cumulative_Volume'], axis=1)
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values
        df = df.fillna(method='ffill')
        
        # If any remaining NaN values (at the start), backward fill
        df = df.fillna(method='bfill')
        
        # If still any NaN values, replace with 0
        df = df.fillna(0)
        
        # Round all float values to 4 decimal places
        for column in df.select_dtypes(include=[np.float64]).columns:
            df[column] = df[column].round(4)
        
        return df
    except Exception as e:
        logging.error(f"Error calculating indicators: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating indicators: {str(e)}")

def calculate_fibonacci_levels(df: pd.DataFrame):
    """Calculate Fibonacci retracement levels based on recent swing highs and lows"""
    lookback_period = min(60, len(df))
    
    swing_high_idx = df['High'].rolling(window=lookback_period).apply(lambda x: x.argmax(), raw=True).iloc[-1]
    swing_high = df.iloc[int(swing_high_idx)]['High']
    swing_high_date = df.iloc[int(swing_high_idx)]['Date']
    
    swing_low_idx = df['Low'].rolling(window=lookback_period).apply(lambda x: x.argmin(), raw=True).iloc[-1]
    swing_low = df.iloc[int(swing_low_idx)]['Low']
    swing_low_date = df.iloc[int(swing_low_idx)]['Date']
    
    if swing_high_date > swing_low_date:
        trend = 'uptrend'
        fib_range = swing_high - swing_low
        base_price = swing_low
    else:
        trend = 'downtrend'
        fib_range = swing_high - swing_low
        base_price = swing_high
    
    fib_levels = {
        '0.0': swing_high if trend == 'downtrend' else swing_low,
        '0.236': base_price + (0.236 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '0.382': base_price + (0.382 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '0.5': base_price + (0.5 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '0.618': base_price + (0.618 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '0.786': base_price + (0.786 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '1.0': swing_low if trend == 'downtrend' else swing_high,
    }
    
    return fib_levels, trend, swing_high, swing_low, swing_high_date, swing_low_date

def calculate_future_projection(df: pd.DataFrame, trend: str, swing_high: float, swing_low: float):
    """Calculate future price projection line"""
    fib_range = swing_high - swing_low
    current_price = df['Close'].iloc[-1]
    
    if trend == 'uptrend':
        future_price = swing_high + 1.618 * fib_range
    else:
        future_price = swing_low - 1.618 * fib_range
    
    future_periods = 30  # Project 30 days into future
    price_diff = future_price - current_price
    slope = price_diff / future_periods
    
    last_date = mdates.num2date(df['Date'].iloc[-1])
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_periods+1)]
    future_prices = [current_price + (i * slope) for i in range(1, future_periods+1)]
    
    future_df = pd.DataFrame({
        'Date': [mdates.date2num(d) for d in future_dates],
        'Projection_Line': future_prices,
        'is_future': True
    })
    
    df['is_future'] = False
    extended_df = pd.concat([df, future_df], ignore_index=True)
    
    return extended_df, slope

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate buy/sell signals with enhanced conditions"""
    try:
        # Initialize Signal column
        df['Signal'] = ""
        
        # Calculate additional indicators safely
        df['Close_SMA_20'] = np.where(
            df['SMA_20'] != 0,
            df['Close'] / df['SMA_20'] - 1,
            0
        )
        
        df['Close_EMA_50'] = np.where(
            df['EMA_50'] != 0,
            df['Close'] / df['EMA_50'] - 1,
            0
        )
        
        # Calculate trend strengths
        df['RSI_Change'] = df['RSI'].diff().fillna(0)
        df['MACD_Change'] = df['MACD'].diff().fillna(0)
        df['Signal_Change'] = df['MACD_Signal'].diff().fillna(0)
        
        # Calculate price momentum
        df['Price_Change'] = df['Close'].pct_change().fillna(0)
        df['Price_Change_5D'] = df['Close'].pct_change(periods=5).fillna(0)
        df['Price_Change_20D'] = df['Close'].pct_change(periods=20).fillna(0)
        
        # Safe volume ratio calculation
        volume_ma = df['Volume'].rolling(20, min_periods=1).mean()
        df['Volume_Ratio'] = np.where(
            volume_ma != 0,
            df['Volume'] / volume_ma,
            1
        )
        
        # Trend Analysis
        df['Trend_Strength'] = 0
        
        # Price trend conditions
        df.loc[df['Close'] > df['SMA_20'], 'Trend_Strength'] += 1
        df.loc[df['Close'] > df['EMA_50'], 'Trend_Strength'] += 1
        df.loc[df['SMA_20'] > df['EMA_50'], 'Trend_Strength'] += 1
        
        # Momentum conditions
        df.loc[df['RSI'] > 50, 'Trend_Strength'] += 1
        df.loc[df['MACD'] > df['MACD_Signal'], 'Trend_Strength'] += 1
        df.loc[df['Price_Change_20D'] > 0, 'Trend_Strength'] += 1
        
        # Define signal conditions
        price_uptrend = (df['Close'] > df['SMA_20']) & (df['Price_Change_5D'] > -0.02)
        price_downtrend = (df['Close'] < df['SMA_20']) & (df['Price_Change_5D'] < 0.02)
        
        rsi_bullish = (df['RSI'] > 45) & (df['RSI'] < 85)  # Wider RSI range for longer periods
        rsi_bearish = (df['RSI'] < 55) & (df['RSI'] > 15)
        
        macd_bullish = (df['MACD'] > df['MACD_Signal']) | (df['MACD_Change'] > 0)
        macd_bearish = (df['MACD'] < df['MACD_Signal']) | (df['MACD_Change'] < 0)
        
        volume_active = df['Volume_Ratio'] > 0.7  # Lower volume threshold for longer periods
        
        # Strong buy conditions
        strong_buy = (
            (df['Trend_Strength'] >= 4) &  # At least 4 trend indicators positive
            price_uptrend &
            rsi_bullish &
            macd_bullish &
            (df['Price_Change_20D'] > -0.01)  # Allow slight decline over 20 days
        )
        
        # Buy conditions
        buy = (
            (df['Trend_Strength'] >= 3) &  # At least 3 trend indicators positive
            price_uptrend &
            (rsi_bullish | macd_bullish) &
            (df['Price_Change_5D'] > -0.02)  # Allow slight decline over 5 days
        )
        
        # Strong sell conditions
        strong_sell = (
            (df['Trend_Strength'] <= 2) &  # At most 2 trend indicators positive
            price_downtrend &
            rsi_bearish &
            macd_bearish &
            (df['Price_Change_20D'] < 0.01)
        )
        
        # Sell conditions
        sell = (
            (df['Trend_Strength'] <= 3) &  # At most 3 trend indicators positive
            price_downtrend &
            (rsi_bearish | macd_bearish) &
            (df['Price_Change_5D'] < 0.02)
        )
        
        # Apply signals
        df.loc[strong_buy, 'Signal'] = "STRONG_BUY"
        df.loc[buy & ~strong_buy, 'Signal'] = "BUY"
        df.loc[strong_sell, 'Signal'] = "STRONG_SELL"
        df.loc[sell & ~strong_sell, 'Signal'] = "SELL"
        
        # Add signal strength and reason
        df['Signal_Strength'] = 0
        df['Signal_Reason'] = ""
        
        for idx in df[df['Signal'] != ""].index:
            reasons = []
            strength = df.loc[idx, 'Trend_Strength']
            
            # Price trend reasons
            if df.loc[idx, 'Close'] > df.loc[idx, 'SMA_20']:
                reasons.append(f"Price > SMA20 ({((df.loc[idx, 'Close']/df.loc[idx, 'SMA_20']-1)*100):.1f}%)")
            if df.loc[idx, 'Close'] > df.loc[idx, 'EMA_50']:
                reasons.append(f"Price > EMA50 ({((df.loc[idx, 'Close']/df.loc[idx, 'EMA_50']-1)*100):.1f}%)")
            
            # RSI reasons
            if 45 <= df.loc[idx, 'RSI'] <= 85:
                reasons.append(f"RSI: {df.loc[idx, 'RSI']:.1f}")
            
            # MACD reasons
            if df.loc[idx, 'MACD'] > df.loc[idx, 'MACD_Signal']:
                reasons.append(f"MACD Bullish ({df.loc[idx, 'MACD']:.2f})")
            
            # Volume reasons
            if df.loc[idx, 'Volume_Ratio'] > 0.7:
                reasons.append(f"Volume: {df.loc[idx, 'Volume_Ratio']:.1f}x Avg")
            
            # Price momentum reasons
            if df.loc[idx, 'Price_Change_5D'] > 0:
                reasons.append(f"5D Change: {(df.loc[idx, 'Price_Change_5D']*100):.1f}%")
            if df.loc[idx, 'Price_Change_20D'] > 0:
                reasons.append(f"20D Change: {(df.loc[idx, 'Price_Change_20D']*100):.1f}%")
            
            df.loc[idx, 'Signal_Strength'] = strength
            df.loc[idx, 'Signal_Reason'] = " | ".join(reasons) if reasons else "Technical Setup"
        
        # Clean up intermediate columns
        df = df.drop(['Close_SMA_20', 'Close_EMA_50', 'RSI_Change', 'MACD_Change', 'Signal_Change',
                     'Volume_Ratio', 'Price_Change', 'Price_Change_5D', 'Price_Change_20D', 'Trend_Strength'], axis=1)
        
        return df
        
    except Exception as e:
        logging.error(f"Error generating signals: {str(e)}")
        return df

def generate_chart_image(df: pd.DataFrame, ticker: str, trend: str, slope: float) -> str:
    """Generate chart image and return as base64 string"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    actual_df = df[~df['is_future']]
    future_df = df[df['is_future']]
    
    # Candlestick plot
    candlestick_ohlc(ax, actual_df[['Date', 'Open', 'High', 'Low', 'Close']].values,
                    width=0.6, colorup='g', colordown='r', alpha=0.8)
    
    # Plot indicators
    ax.plot(actual_df['Date'], actual_df['SMA_20'], 'b-', label='SMA 20', alpha=0.7)
    ax.plot(actual_df['Date'], actual_df['EMA_50'], 'm-', label='EMA 50', alpha=0.7)
    
    # Plot signals
    for idx, row in actual_df[actual_df['Signal'] != ""].iterrows():
        y_pos = row['Low'] * 0.98 if row['Signal'] == "BUY" else row['High'] * 1.02
        color = 'green' if row['Signal'] == "BUY" else 'red'
        ax.text(row['Date'], y_pos, row['Signal'], 
               fontsize=10, weight='bold', color=color,
               ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor=color))
    
    # Future projection line
    ax.plot(future_df['Date'], future_df['Projection_Line'], 'g--', linewidth=2, alpha=0.8, label='Future Projection')
    
    # Chart formatting
    ax.text(0.02, 0.95, f"Trend: {trend.capitalize()} | Projection Slope: {slope:.4f}",
           transform=ax.transAxes, fontsize=10,
           bbox=dict(facecolor='white', alpha=0.8))
    
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f'{ticker} - Daily Trading Signals', fontsize=14)
    ax.set_ylabel('Price (₹)')
    ax.legend(loc='upper left')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    # Convert to base64 string
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

# Add new LSTM Prediction endpoints
class LSTMRequest(BaseModel):
    ticker: str
    period: Optional[str] = "2y"
    prediction_days: Optional[int] = 20
    anomaly_threshold: Optional[float] = 2.5

@app.post("/api/lstm/predict")
async def get_lstm_prediction(request: LSTMRequest):
    """Get LSTM-based stock price prediction"""
    try:
        # Fetch and process data
        df = fetch_stock_data(request.ticker, request.period)
        df = calculate_technical_indicators(df)
        df = detect_anomalies(df, threshold=request.anomaly_threshold)
        df = generate_signals(df)
        fib_levels = calculate_fibonacci_levels(df)
        
        # Prepare and train LSTM model
        sequence_length = 60
        X, y, scaler = prepare_data(df, sequence_length)
        model = create_lstm_model(sequence_length)
        
        # Train model
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        # Make predictions
        last_sequence = X[-1]
        predictions = predict_future_prices(model, last_sequence, scaler, request.prediction_days)
        
        # Prepare prediction data
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=x+1) for x in range(request.prediction_days)]
        current_price = df['Close'].iloc[-1]
        
        prediction_data = []
        for i, (date, pred_price) in enumerate(zip(future_dates, predictions)):
            daily_change = ((pred_price[0] - current_price) / current_price) * 100 if i == 0 else \
                         ((pred_price[0] - predictions[i-1][0]) / predictions[i-1][0]) * 100
            prediction_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_price': float(pred_price[0]),
                'daily_change': float(daily_change),
                'cumulative_change': float(((pred_price[0] - current_price) / current_price) * 100)
            })
        
        # Get technical analysis summary
        current_signal = df['Signal'].iloc[-1]
        signal_text = "BUY" if current_signal == 1 else "SELL" if current_signal == -1 else "NEUTRAL"
        
        # Get recent anomalies
        recent_anomalies = df[df['Is_Anomaly']].tail(5)
        anomaly_data = []
        if not recent_anomalies.empty:
            for idx, row in recent_anomalies.iterrows():
                anomaly_data.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'price': float(row['Close']),
                    'z_score': float(row['Z_Score']),
                    'price_change_pct': float(row['Price_Change_Pct']),
                    'volume_ratio': float(row['Volume_Ratio'])
                })
        
        return {
            "status": "success",
            "ticker": request.ticker,
            "current_price": float(current_price),
            "predictions": prediction_data,
            "technical_analysis": {
                "current_signal": signal_text,
                "rsi": float(df['RSI'].iloc[-1]),
                "macd": float(df['MACD'].iloc[-1]),
                "macd_signal": float(df['MACD_Signal'].iloc[-1]),
                "sma_20": float(df['SMA_20'].iloc[-1]),
                "sma_50": float(df['SMA_50'].iloc[-1])
            },
            "anomalies": anomaly_data,
            "fibonacci_levels": {k: float(v) for k, v in fib_levels.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/lstm/technical-indicators/{symbol}")
async def get_lstm_technical_indicators(symbol: str, period: str = "2y"):
    """Get technical indicators for LSTM analysis"""
    try:
        df = fetch_stock_data(symbol, period)
        df = calculate_technical_indicators(df)
        
        # Get the latest values
        latest_data = {
            "symbol": symbol,
            "indicators": {
                "sma_20": float(df['SMA_20'].iloc[-1]),
                "sma_50": float(df['SMA_50'].iloc[-1]),
                "sma_200": float(df['SMA_200'].iloc[-1]),
                "macd": float(df['MACD'].iloc[-1]),
                "macd_signal": float(df['MACD_Signal'].iloc[-1]),
                "macd_hist": float(df['MACD_Hist'].iloc[-1]),
                "rsi": float(df['RSI'].iloc[-1]),
                "bb_upper": float(df['BB_Upper'].iloc[-1]),
                "bb_middle": float(df['BB_Middle'].iloc[-1]),
                "bb_lower": float(df['BB_Lower'].iloc[-1])
            },
            "price_data": {
                "current_price": float(df['Close'].iloc[-1]),
                "open": float(df['Open'].iloc[-1]),
                "high": float(df['High'].iloc[-1]),
                "low": float(df['Low'].iloc[-1]),
                "volume": float(df['Volume'].iloc[-1])
            }
        }
        
        return {
            "status": "success",
            "data": latest_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/lstm/anomalies/{symbol}")
async def get_lstm_anomalies(symbol: str, period: str = "2y", threshold: float = 2.5):
    """Get detected price anomalies"""
    try:
        df = fetch_stock_data(symbol, period)
        df = detect_anomalies(df, threshold=threshold)
        
        anomalies = df[df['Is_Anomaly']]
        anomaly_data = []
        
        for idx, row in anomalies.iterrows():
            anomaly_data.append({
                'date': idx.strftime('%Y-%m-%d'),
                'price': float(row['Close']),
                'z_score': float(row['Z_Score']),
                'price_change_pct': float(row['Price_Change_Pct']),
                'volume_ratio': float(row['Volume_Ratio'])
            })
        
        return {
            "status": "success",
            "symbol": symbol,
            "anomalies": anomaly_data,
            "total_anomalies": len(anomaly_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\nShutting down gracefully...")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the server with proper shutdown handling
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        timeout_keep_alive=30,
        loop="asyncio"
    ) 