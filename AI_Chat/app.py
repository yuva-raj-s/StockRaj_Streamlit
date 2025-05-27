import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import json
import os
import sys
from textblob import TextBlob
import logging
from transformers import pipeline
import torch
import numpy as np
from fuzzywuzzy import process
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import re
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib
from GoogleNews import GoogleNews
matplotlib.use('Agg')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from chatbot import IndianStockChatbot as MLChatbot  # Import using absolute path

# Add Portfolio and Watchlist directories to sys.path for import
portfolio_dir = os.path.abspath(os.path.join(current_dir, '../Portfolio'))
watchlist_dir = os.path.abspath(os.path.join(current_dir, '../Watchlist'))
if portfolio_dir not in sys.path:
    sys.path.append(portfolio_dir)
if watchlist_dir not in sys.path:
    sys.path.append(watchlist_dir)

from portfolio import load_portfolio, calculate_portfolio_metrics
from watchlist_operations import WatchlistManager

class IndianStockChatbot:
    def __init__(self):
        """Initialize the chatbot with ML capabilities"""
        try:
            print("Initializing chatbot...")
            
            # Initialize models
            self.models = {
                'intent': pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english"),
                'sentiment': pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"),
                'text_qa': pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            }
            
            # Initialize prediction model
            self.prediction_model = self._initialize_prediction_model()
            
            # Load stock symbols mapping
            self.stock_symbols = self._load_stock_symbols()
            
            # Load market terms
            self.market_terms = self._load_market_terms()
            
            # Load intent patterns
            self.intent_patterns = self._load_intent_patterns()
            
            # Initialize history
            self.history = []
            
            # Initialize portfolio and watchlist managers
            self.watchlist_manager = WatchlistManager()
            
            # Ensure portfolio and watchlist directories exist
            os.makedirs("Portfolio", exist_ok=True)
            os.makedirs("Watchlist", exist_ok=True)
            
            print("Chatbot initialization completed!")
            
        except Exception as e:
            logging.error(f"Error initializing chatbot: {str(e)}")
            print(f"Error loading models: {str(e)}")
            # Initialize with minimal functionality
            self.models = {
                'intent': lambda x: [{'label': 'general_query', 'score': 1.0}],
                'sentiment': lambda x: [{'label': 'neutral', 'score': 1.0}],
                'text_qa': lambda x: x[:200] + "..."
            }
            self.prediction_model = None
            self.history = []
            self.stock_symbols = {}
            self.market_terms = {}
            self.intent_patterns = {}
            self.watchlist_manager = None

    def _load_stock_symbols(self):
        """Load stock symbols mapping"""
        return {
            "reliance": "RELIANCE",
            "reliance industries": "RELIANCE",
            "ril": "RELIANCE",
            "tcs": "TCS",
            "tata consultancy": "TCS",
            "tata consultancy services": "TCS",
            "infosys": "INFY",
            "infy": "INFY",
            "hdfc bank": "HDFCBANK",
            "hdfc": "HDFCBANK",
            "icici bank": "ICICIBANK",
            "icici": "ICICIBANK",
            "wipro": "WIPRO",
            "tata motors": "TATAMOTORS",
            "tatamotors": "TATAMOTORS",
            "tata steel": "TATASTEEL",
            "tatasteel": "TATASTEEL",
            "bharti airtel": "BHARTIARTL",
            "airtel": "BHARTIARTL",
            "sbi": "SBIN",
            "state bank": "SBIN",
            "state bank of india": "SBIN",
            "axis bank": "AXISBANK",
            "axis": "AXISBANK",
            "kotak bank": "KOTAKBANK",
            "kotak": "KOTAKBANK",
            "asian paints": "ASIANPAINT",
            "asian": "ASIANPAINT",
            "bajaj auto": "BAJAJ-AUTO",
            "bajaj": "BAJAJ-AUTO",
            "hindalco": "HINDALCO",
            "itc": "ITC",
            "larsen": "LT",
            "l&t": "LT",
            "larsen and toubro": "LT",
            "m&m": "M&M",
            "mahindra": "M&M",
            "maruti": "MARUTI",
            "maruti suzuki": "MARUTI",
            "nestle": "NESTLEIND",
            "nestle india": "NESTLEIND",
            "ongc": "ONGC",
            "oil and natural gas": "ONGC",
            "power grid": "POWERGRID",
            "sun pharma": "SUNPHARMA",
            "sun": "SUNPHARMA",
            "titan": "TITAN",
            "ultracemco": "ULTRACEMCO",
            "ultra cement": "ULTRACEMCO"
        }

    def _load_market_terms(self):
        """Load market terms and their explanations"""
        return {
            "nifty": "Nifty 50 is an index of 50 major companies listed on the National Stock Exchange (NSE) of India.",
            "sensex": "Sensex is an index of 30 major companies listed on the Bombay Stock Exchange (BSE) of India.",
            "ipo": "Initial Public Offering (IPO) is when a private company offers its shares to the public for the first time.",
            "fii": "Foreign Institutional Investors (FIIs) are entities that invest in Indian markets from outside India.",
            "dii": "Domestic Institutional Investors (DIIs) are Indian entities that invest in the stock market.",
            "circuit": "Circuit limits are the maximum percentage by which a stock can move up or down in a single day.",
            "upper circuit": "Upper circuit is the maximum percentage a stock can rise in a single day.",
            "lower circuit": "Lower circuit is the maximum percentage a stock can fall in a single day.",
            "market": "The Indian stock market consists of two major exchanges: NSE and BSE. Trading hours are 9:15 AM to 3:30 PM IST on weekdays.",
            "trading": "Trading in Indian stocks happens on NSE and BSE from 9:15 AM to 3:30 PM IST on weekdays.",
            "invest": "Investing in Indian stocks requires a demat account and trading account. You can invest through brokers or online platforms.",
            "broker": "Stock brokers in India are regulated by SEBI. Popular brokers include Zerodha, ICICI Direct, HDFC Securities, and Kotak Securities.",
            "demat": "A Demat account is required to hold shares in electronic form. It's mandatory for trading in Indian stocks.",
            "sebi": "SEBI (Securities and Exchange Board of India) is the regulator for securities markets in India.",
            "sector": "Major sectors in Indian markets include IT, Banking, FMCG, Auto, Pharma, and Infrastructure.",
            "dividend": "Dividends are payments made by companies to their shareholders from profits. They are usually paid quarterly or annually.",
            "mutual fund": "Mutual funds pool money from investors to invest in stocks, bonds, and other securities. They are managed by professional fund managers.",
            "etf": "ETFs (Exchange Traded Funds) are investment funds that track an index and trade like stocks on exchanges."
        }

    def _load_intent_patterns(self):
        """Load intent patterns for better understanding"""
        return {
            'price_query': [
                'price', 'current price', 'stock price', 'share price', 'value',
                'how much', 'what is the price', 'current value'
            ],
            'market_status': [
                'market status', 'market open', 'trading hours', 'market timing',
                'is market open', 'when does market open'
            ],
            'index_query': [
                'nifty', 'sensex', 'index', 'market index', 'benchmark',
                'nifty 50', 'bse sensex'
            ],
            'term_query': [
                'what is', 'explain', 'define', 'meaning of', 'tell me about',
                'ipo', 'fii', 'dii', 'circuit', 'demat', 'sebi'
            ],
            'analysis_query': [
                'analysis', 'outlook', 'trend', 'performance', 'how is',
                'what about', 'tell me about'
            ]
        }

    def _initialize_prediction_model(self):
        """Initialize the LSTM prediction model"""
        try:
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(60, 5)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model
        except Exception as e:
            logging.error(f"Error initializing prediction model: {str(e)}")
            return None

    def get_market_status(self):
        """Get current market status using real-time data"""
        try:
            now = datetime.now()
            
            # Check if it's a weekend
            if now.weekday() >= 5:  # Saturday (5) or Sunday (6)
                return "Closed"
            
            # Define market hours
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            # Get real-time NSE data to confirm market status
            nifty = yf.Ticker("^NSEI")
            nifty_data = nifty.history(period="1d", interval="1m")
            
            if not nifty_data.empty:
                last_update = nifty_data.index[-1]
                time_diff = (now - last_update).total_seconds() / 60
                
                # If we have recent data (within last 5 minutes) and within market hours
                if time_diff <= 5 and market_open <= now <= market_close:
                    return "Open"
                else:
                    return "Closed"
            
            # Fallback to time-based check
            return "Open" if market_open <= now <= market_close else "Closed"
            
        except Exception as e:
            logging.error(f"Error getting market status: {str(e)}")
            # Fallback to time-based check if there's an error
            try:
                now = datetime.now()
                market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
                return "Open" if market_open <= now <= market_close else "Closed"
            except:
                return "Unknown"

    def get_stock_info(self, symbol):
        """Get stock information"""
        try:
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if info and 'regularMarketPrice' in info:
                return {
                    'symbol': symbol,
                    'name': info.get('shortName', symbol),
                    'price': info['regularMarketPrice'],
                    'change': info.get('regularMarketChange', 0),
                    'change_percent': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('regularMarketVolume', 0),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 'N/A'),
                    'dividend_yield': info.get('dividendYield', 0)
                }
            return None
        except Exception as e:
            logging.error(f"Error getting stock info: {str(e)}")
            return None

    def get_market_activity(self):
        """Get real-time market activity data"""
        try:
            # Get Nifty 50 data with real-time updates
            nifty = yf.Ticker("^NSEI")
            nifty_data = nifty.history(period="1d", interval="1m")
            
            # Get Sensex data with real-time updates
            sensex = yf.Ticker("^BSESN")
            sensex_data = sensex.history(period="1d", interval="1m")
            
            if not nifty_data.empty and not sensex_data.empty:
                # Get the latest data points
                nifty_latest = nifty_data.iloc[-1]
                sensex_latest = sensex_data.iloc[-1]
                
                # Calculate changes using the latest data
                nifty_change = nifty_latest['Close'] - nifty_data.iloc[0]['Open']
                nifty_change_pct = (nifty_change / nifty_data.iloc[0]['Open']) * 100
                
                sensex_change = sensex_latest['Close'] - sensex_data.iloc[0]['Open']
                sensex_change_pct = (sensex_change / sensex_data.iloc[0]['Open']) * 100
                
                return {
                    'nifty': {
                        'current': nifty_latest['Close'],
                        'change_pct': nifty_change_pct,
                        'high': nifty_data['High'].max(),
                        'low': nifty_data['Low'].min(),
                        'volume': nifty_latest['Volume'],
                        'last_update': nifty_data.index[-1].strftime("%H:%M:%S")
                    },
                    'sensex': {
                        'current': sensex_latest['Close'],
                        'change_pct': sensex_change_pct,
                        'high': sensex_data['High'].max(),
                        'low': sensex_data['Low'].min(),
                        'volume': sensex_latest['Volume'],
                        'last_update': sensex_data.index[-1].strftime("%H:%M:%S")
                    },
                    'market_status': self.get_market_status(),
                    'advance_decline': self.get_advance_decline_ratio(),
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            return None
        except Exception as e:
            logging.error(f"Error getting market activity: {str(e)}")
            return None

    def get_advance_decline_ratio(self):
        """Get advance-decline ratio"""
        try:
            # Get NSE data
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="1d", interval="1m")
            
            if not hist.empty:
                advances = len(hist[hist['Close'] > hist['Open']])
                declines = len(hist[hist['Close'] < hist['Open']])
                unchanged = len(hist) - advances - declines
                
                return {
                    'advances': advances,
                    'declines': declines,
                    'unchanged': unchanged,
                    'ratio': advances / declines if declines > 0 else float('inf')
                }
            return {'advances': 0, 'declines': 0, 'unchanged': 0, 'ratio': 0}
        except Exception as e:
            logging.error(f"Error getting advance-decline ratio: {str(e)}")
            return {'advances': 0, 'declines': 0, 'unchanged': 0, 'ratio': 0}

    def get_sentiment(self, symbol):
        """Get sentiment analysis for a stock"""
        try:
            logging.info(f"Starting sentiment analysis for {symbol}")
            
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            # Get stock info first
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get('regularMarketPrice', 0)
                price_change = info.get('regularMarketChangePercent', 0)
                logging.info(f"Got stock info - Price: {current_price}, Change: {price_change}")
            except Exception as e:
                logging.error(f"Error getting stock info: {str(e)}")
                return None

            # Fetch news from Google News
            try:
                logging.info("Fetching news from Google News")
                googlenews = GoogleNews(lang="en")
                company_name = symbol.replace('.NS', '')
                googlenews.search(company_name)
                news = googlenews.result()
                
                # Get more pages if available
                page = 2
                while len(news) < 10 and page <= 3:  # Try up to 3 pages
                    googlenews.get_page(page)
                    page_results = googlenews.result()
                    if not page_results:
                        break
                    news.extend(page_results)
                    page += 1
                
                news = news[:10]  # Keep only top 10 articles
                logging.info(f"Got {len(news)} news articles from Google News")
            except Exception as e:
                logging.error(f"Error getting Google News: {str(e)}")
                return None

            if not news:
                logging.warning("No news articles found")
                return None

            # Analyze news sentiment using the financial model
            sentiments = []
            positive_news = []
            negative_news = []
            neutral_news = []
            
            for article in news:
                try:
                    if 'title' in article:
                        # Combine title and description for better context
                        text = f"{article['title']} {article.get('desc', '')}"
                        
                        # Get sentiment using the financial model
                        try:
                            sentiment = self.models['sentiment'](text)[0]
                            sentiment_score = sentiment['score']
                            sentiment_label = sentiment['label']
                            logging.info(f"Analyzed sentiment for article: {article['title'][:50]}... - Score: {sentiment_score}, Label: {sentiment_label}")
                        except Exception as e:
                            logging.error(f"Error in sentiment analysis: {str(e)}")
                            continue

                        # Calculate time weight
                        time_weight = self._calculate_time_weight(article.get('date', ''))
                        
                        # Calculate base score and weighted addition
                        base_score = 3 if sentiment_label == 'positive' else -3 if sentiment_label == 'negative' else 0
                        weighted_addition = base_score * time_weight
                        total_score = base_score + weighted_addition
                        
                        # Categorize news with time weight
                        news_item = {
                            'title': article['title'],
                            'source': article.get('publisher', 'Unknown'),
                            'date': article.get('date', 'Unknown'),
                            'sentiment': sentiment_score,
                            'weight': time_weight,
                            'base_score': base_score,
                            'weighted_addition': weighted_addition,
                            'total_score': total_score
                        }
                        
                        if sentiment_label == 'positive':
                            positive_news.append(news_item)
                        elif sentiment_label == 'negative':
                            negative_news.append(news_item)
                        else:
                            neutral_news.append(news_item)
                        
                        sentiments.append(total_score)
                except Exception as e:
                    logging.error(f"Error processing article: {str(e)}")
                    continue

            if sentiments:
                # Calculate weighted average sentiment
                avg_sentiment = sum(sentiments) / len(sentiments)
                
                # Determine overall sentiment label
                if avg_sentiment > 1:
                    sentiment_label = "Positive"
                elif avg_sentiment < -1:
                    sentiment_label = "Negative"
                else:
                    sentiment_label = "Neutral"
                
                logging.info(f"Sentiment analysis complete - Average: {avg_sentiment:.2f}, Label: {sentiment_label}")
                
                return {
                    'current_price': current_price,
                    'price_change': price_change,
                    'average': avg_sentiment,
                    'label': sentiment_label,
                    'positive_count': len(positive_news),
                    'negative_count': len(negative_news),
                    'neutral_count': len(neutral_news),
                    'total_news': len(news)
                }
            
            logging.warning("No valid sentiments calculated")
            return None
            
        except Exception as e:
            logging.error(f"Error in get_sentiment: {str(e)}")
            return None

    def _calculate_time_weight(self, article_date_str):
        """Calculate time weight for news articles"""
        try:
            date_formats = [
                '%a, %d %b %Y %H:%M:%S %z',
                '%Y-%m-%d %H:%M:%S',
                '%a, %d %b %Y %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S%z',
                '%a %b %d, %Y',
                '%d %b %Y'
            ]
            
            parsed_date = None
            for format_str in date_formats:
                try:
                    parsed_date = datetime.strptime(article_date_str, format_str)
                    break
                except ValueError:
                    continue
            
            if parsed_date is None:
                return 0.01
            
            now = datetime.now()
            if parsed_date.tzinfo is not None:
                now = now.replace(tzinfo=parsed_date.tzinfo)
            
            hours_diff = (now - parsed_date).total_seconds() / 3600
            
            if hours_diff < 1:  # Within last hour
                return 0.24
            elif hours_diff < 24:  # Within last 24 hours
                return max(0.01, 0.24 - ((hours_diff - 1) * 0.01))
            else:
                return 0.01
        except Exception as e:
            logging.error(f"Error calculating time weight: {e}")
            return 0.01

    def _create_sentiment_chart(self, positive_count, neutral_count, negative_count):
        """Create sentiment distribution chart"""
        try:
            total = positive_count + neutral_count + negative_count
            if total == 0:
                return None
            
            positive_pct = (positive_count / total) * 100
            neutral_pct = (neutral_count / total) * 100
            negative_pct = (negative_count / total) * 100
            
            fig, ax = plt.subplots(figsize=(5, 0.5))
            ax.barh([0], positive_pct, color='#28a745', label='Positive')
            ax.barh([0], neutral_pct, left=positive_pct, color='#6c757d', label='Neutral')
            ax.barh([0], negative_pct, left=positive_pct+neutral_pct, color='#dc3545', label='Negative')
            ax.set_xlim(0, 100)
            ax.axis('off')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf)
        except Exception as e:
            logging.error(f"Error creating sentiment chart: {e}")
            return None

    def process_query(self, query):
        """Process user query and generate response"""
        try:
            # Add to history
            self.history.append(query)
            
            # Clean and normalize the query
            cleaned_query = self.clean_query(query)
            
            # Check for sentiment analysis queries
            if 'sentiment' in cleaned_query.lower():
                symbol = self.get_stock_symbol(cleaned_query)
                if symbol:
                    logging.info(f"Processing sentiment query for symbol: {symbol}")
                    data = self.get_sentiment(symbol)
                    if data:
                        response = f"📊 Sentiment Analysis for {symbol}:\n\n"
                        response += f"Current Price: ₹{data['current_price']:,.2f} ({data['price_change']:+.2f}%)\n"
                        response += f"Overall Sentiment: {data['label']}\n"
                        response += f"Sentiment Score: {data['average']:.2f}\n"
                        return response
                    logging.warning(f"No sentiment data available for {symbol}")
                    return f"Unable to fetch sentiment analysis for {symbol} at the moment. Please try again later."
                return "Please specify which stock's sentiment you'd like to know about."
            
            # Check for watchlist queries
            if any(phrase in cleaned_query.lower() for phrase in ['watchlist', 'show watchlist', 'my watchlist']):
                try:
                    watchlist_manager = WatchlistManager()
                    watchlist_data = watchlist_manager.get_watchlist_data()
                    
                    if not watchlist_data.empty:
                        response = "📋 Your Watchlist:\n\n"
                        for _, row in watchlist_data.iterrows():
                            change_emoji = "🟢" if float(row['Change %'].strip('%')) > 0 else "🔴" if float(row['Change %'].strip('%')) < 0 else "⚪"
                            response += f"{change_emoji} {row['Symbol']}\n"
                            response += f"   Price: {row['Current Price']}\n"
                            response += f"   Change: {row['Change']} ({row['Change %']})\n"
                            response += f"   Day Range: {row['Day Low']} - {row['Day High']}\n"
                            response += f"   Market Cap: {row['Market Cap']}\n\n"
                        
                        # Add watchlist summary
                        total_stocks = len(watchlist_data)
                        up_stocks = len(watchlist_data[watchlist_data['Change %'].str.strip('%').astype(float) > 0])
                        down_stocks = len(watchlist_data[watchlist_data['Change %'].str.strip('%').astype(float) < 0])
                        
                        response += f"\n📊 Watchlist Summary:\n"
                        response += f"Total Stocks: {total_stocks}\n"
                        response += f"Up: {up_stocks} | Down: {down_stocks} | Unchanged: {total_stocks - up_stocks - down_stocks}\n"
                        return response
                    return "Your watchlist is empty. Add stocks to your watchlist to track them."
                except Exception as e:
                    logging.error(f"Error processing watchlist query: {str(e)}")
                    return "Unable to fetch watchlist data at the moment."
            
            # Check for portfolio queries
            if any(phrase in cleaned_query.lower() for phrase in ['portfolio', 'show portfolio', 'my portfolio']):
                try:
                    portfolio_data = load_portfolio()
                    if portfolio_data and portfolio_data.get('holdings'):
                        metrics = calculate_portfolio_metrics(portfolio_data)
                        if metrics:
                            response = "📊 Your Portfolio:\n\n"
                            
                            # Individual holdings
                            for holding in metrics['holdings']:
                                profit_emoji = "🟢" if holding['pnl'] > 0 else "🔴" if holding['pnl'] < 0 else "⚪"
                                response += f"{profit_emoji} {holding['symbol']}\n"
                                response += f"   Quantity: {holding['quantity']:,}\n"
                                response += f"   Avg Price: ₹{holding['avg_price']:,.2f}\n"
                                response += f"   Current Price: ₹{holding['current_price']:,.2f}\n"
                                response += f"   Current Value: ₹{holding['current_value']:,.2f}\n"
                                response += f"   P/L: ₹{holding['pnl']:,.2f} ({holding['pnl_percent']:+.2f}%)\n\n"
                            
                            # Portfolio summary
                            response += "📈 Portfolio Summary:\n"
                            response += f"Total Investment: ₹{metrics['total_invested']:,.2f}\n"
                            response += f"Current Value: ₹{metrics['total_value']:,.2f}\n"
                            response += f"Total P/L: ₹{metrics['total_pnl']:,.2f} ({metrics['total_pnl_percent']:+.2f}%)\n"
                            
                            if metrics.get('skipped_symbols'):
                                response += f"\n⚠️ Note: Could not fetch data for: {', '.join(metrics['skipped_symbols'])}"
                            
                            return response
                    return "Your portfolio is empty. Add stocks to your portfolio to track your investments."
                except Exception as e:
                    logging.error(f"Error processing portfolio query: {str(e)}")
                    return "Unable to fetch portfolio data at the moment."
            
            # Check for market activity queries
            if any(phrase in cleaned_query.lower() for phrase in ['market activity', 'market status', 'market overview', 'show market']):
                data = self.get_market_activity()
                if data:
                    response = "Market Activity Overview:\n"
                    response += f"\nNifty 50: ₹{data['nifty']['current']:.2f} ({data['nifty']['change_pct']:+.2f}%)"
                    response += f"\nHigh: ₹{data['nifty']['high']:.2f}, Low: ₹{data['nifty']['low']:.2f}"
                    response += f"\nSensex: ₹{data['sensex']['current']:.2f} ({data['sensex']['change_pct']:+.2f}%)"
                    response += f"\nHigh: ₹{data['sensex']['high']:.2f}, Low: ₹{data['sensex']['low']:.2f}"
                    
                    if data.get('sector_performance'):
                        response += "\n\nSector Performance:"
                        for sector, perf in data['sector_performance'].items():
                            response += f"\n{sector}: {perf['change_pct']:+.2f}%"
                    
                    if data['advance_decline']['ratio'] != float('inf'):
                        response += f"\n\nAdvance-Decline Ratio: {data['advance_decline']['ratio']:.2f}"
                    
                    response += f"\n\nLast Updated: {data['last_updated']}"
                    return response
                return "Unable to fetch market activity at the moment."
            
            # Check for stock price queries
            if any(word in cleaned_query.lower() for word in ['price', 'value', 'current price']):
                symbol = self.get_stock_symbol(cleaned_query)
                if symbol:
                    data = self.get_stock_info(symbol)
                    if data:
                        response = f"Stock Information for {data['name']} ({data['symbol']}):\n"
                        response += f"Current Price: ₹{data['price']:.2f}\n"
                        response += f"Change: {data['change']:+.2f} ({data['change_percent']:+.2f}%)\n"
                        response += f"Volume: {data['volume']:,}\n"
                        response += f"Market Cap: ₹{data['market_cap']/10000000:.2f} Cr\n"
                        response += f"P/E Ratio: {data['pe_ratio']}\n"
                        response += f"Dividend Yield: {data['dividend_yield']*100:.2f}%"
                        return response
                    return f"Unable to fetch stock information for {symbol} at the moment."
                return "Please specify which stock's price you'd like to know about."
            
            # Check for market term queries
            for term, explanation in self.market_terms.items():
                if term in cleaned_query.lower():
                    return f"{term.upper()}: {explanation}"
            
            # If no specific query type is matched, return a general response
            return "I can help you with stock prices, market activity, sentiment analysis, portfolio, watchlist, and market terms. Please specify what information you're looking for."
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return "I'm having trouble understanding. Could you please rephrase your question?"

    def clean_query(self, text):
        """Clean and normalize the query text"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s.]', '', text)
        text = ' '.join(text.split())
        return text

    def get_stock_symbol(self, user_input):
        """Get stock symbol from user input"""
        try:
            # First try exact match from predefined symbols
            for key, symbol in self.stock_symbols.items():
                if key in user_input.lower():
                    return symbol
            
            # Try fuzzy matching with predefined symbols
            match, score = process.extractOne(user_input.lower(), self.stock_symbols.keys())
            if score > 80:
                return self.stock_symbols[match]
            
            # Try dynamic lookup using yfinance
            try:
                # Remove common words and clean the input
                clean_input = re.sub(r'\b(stock|share|price|value|current|latest)\b', '', user_input.lower())
                clean_input = clean_input.strip()
                
                # Try to get the stock info
                ticker = yf.Ticker(f"{clean_input}.NS")
                info = ticker.info
                
                if info and 'symbol' in info:
                    # Add to stock_symbols for future use
                    self.stock_symbols[clean_input] = info['symbol']
                    return info['symbol']
            except:
                pass
            
            return None
        except Exception as e:
            logging.error(f"Error getting stock symbol: {str(e)}")
            return None

    def _load_portfolio(self):
        """Load portfolio data"""
        try:
            if os.path.exists("Portfolio/portfolio_data.json"):
                with open("Portfolio/portfolio_data.json", "r") as f:
                    return json.load(f)
            return {"holdings": {}, "transactions": []}
        except Exception as e:
            logging.error(f"Error loading portfolio: {str(e)}")
            return {"holdings": {}, "transactions": []}

    def _load_watchlist(self):
        """Load watchlist data"""
        try:
            if os.path.exists("Watchlist/watchlist.json"):
                with open("Watchlist/watchlist.json", "r") as f:
                    return json.load(f)
            return []
        except Exception as e:
            logging.error(f"Error loading watchlist: {str(e)}")
            return []

def show_chat():
    """Display the AI chat page"""
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = [
            "What's the price of TCS?",
            "Show market activity",
            "What's the sentiment for Reliance?",
            "Show my watchlist",
            "Show my portfolio"
        ]

    # Custom CSS
    st.markdown("""
    <style>
        body, .main, .block-container {
            background-color: #101820 !important;
        }
        .stApp {
            background-color: #101820 !important;
        }
        .stTextInput>div>div>input {
            font-size: 1.1rem;
            background-color: #1B2C3A !important;
            color: #fff !important;
            border: 1.5px solid #223344 !important;
        }
        .chat-outer {
            max-height: 480px;
            min-height: 320px;
            overflow-y: auto;
            padding-bottom: 1rem;
            margin-bottom: 1.2rem;
            border-radius: 1.2rem;
            background: rgba(27,44,58,0.13);
            box-shadow: 0 2px 12px rgba(27,44,58,0.10);
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 2rem;
            margin-bottom: 0.5rem;
        }
        .chat-message {
            display: flex;
            align-items: flex-end;
            gap: 1.2rem;
            max-width: 80%;
            border-radius: 1.7rem;
            box-shadow: 0 4px 24px rgba(27,44,58,0.13);
            padding: 1.5rem 1.7rem;
            margin-bottom: 0.7rem;
            font-size: 1.13rem;
            word-break: break-word;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            opacity: 1;
            transition: box-shadow 0.2s, opacity 0.4s;
        }
        .chat-message.bot.new {
            animation: fadeInBot 0.7s;
        }
        @keyframes fadeInBot {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .chat-message.user {
            background: linear-gradient(135deg, #1B2C3A 85%, #223344 100%);
            color: #fff;
            margin-left: auto;
            border-bottom-right-radius: 0.7rem;
            border-top-right-radius: 0.7rem;
            border-bottom-left-radius: 1.7rem;
            border-top-left-radius: 1.7rem;
            border: 2px solid #223344;
        }
        .chat-message.bot {
            background: linear-gradient(135deg, #223344 85%, #1B2C3A 100%);
            color: #e6eaf0;
            margin-right: auto;
            border-bottom-left-radius: 0.7rem;
            border-top-left-radius: 0.7rem;
            border-bottom-right-radius: 1.7rem;
            border-top-right-radius: 1.7rem;
            border: 2px solid #1B2C3A;
        }
        .chat-avatar {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.7rem;
            font-weight: bold;
            box-shadow: 0 2px 8px rgba(27,44,58,0.13);
        }
        .chat-avatar.user {
            background: #1B2C3A;
            color: #fff;
            border: 2.5px solid #223344;
        }
        .chat-avatar.bot {
            background: #223344;
            color: #fff;
            border: 2.5px solid #1B2C3A;
        }
        .chat-message .content {
            margin-top: 0;
            flex: 1;
        }
        .sticky-input {
            position: sticky;
            bottom: 0;
            background: #101820;
            z-index: 10;
            padding-top: 1rem;
            padding-bottom: 0.5rem;
        }
        .input-separator {
            border: none;
            border-top: 2px solid #223344;
            margin: 0.5rem 0 1.2rem 0;
        }
        .loading-bot {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin-left: 2.5rem;
            margin-bottom: 1.2rem;
        }
        .loading-dots span {
            display: inline-block;
            width: 10px;
            height: 10px;
            margin-right: 2px;
            background: #e6eaf0;
            border-radius: 50%;
            opacity: 0.7;
            animation: loadingFade 1.2s infinite;
        }
        .loading-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }
        .loading-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }
        @keyframes loadingFade {
            0%, 80%, 100% { opacity: 0.3; }
            40% { opacity: 1; }
        }
        .stButton>button {
            width: 100%;
            border-radius: 0.7rem;
            height: 3.2rem;
            font-size: 1.13rem;
            background-color: #1B2C3A !important;
            color: #fff !important;
            border: 1.5px solid #223344 !important;
            font-weight: 600;
            transition: background 0.2s, color 0.2s;
        }
        .stButton>button:hover {
            background-color: #223344 !important;
            color: #fff !important;
        }
        .suggestion-chip {
            display: inline-block;
            padding: 0.6rem 1.2rem;
            margin: 0.3rem;
            border-radius: 1.2rem;
            background-color: #1B2C3A;
            color: #fff;
            cursor: pointer;
            border: 1.5px solid #223344;
            font-size: 1.05rem;
            font-weight: 500;
            box-shadow: 0 2px 8px rgba(27,44,58,0.10);
        }
        .suggestion-chip:hover {
            background-color: #223344;
            color: #fff;
        }
        .stMetric {
            background-color: #1B2C3A !important;
            color: #fff !important;
            border-radius: 0.7rem;
            padding: 0.7rem 1.2rem;
            box-shadow: 0 2px 8px rgba(27,44,58,0.10);
        }
        .stExpander, .stExpanderHeader {
            background-color: #1B2C3A !important;
            color: #fff !important;
            border-radius: 0.7rem !important;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #fff !important;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .section-card {
            background: linear-gradient(135deg, #1B2C3A 90%, #223344 100%);
            border-radius: 1.2rem;
            box-shadow: 0 4px 24px rgba(27,44,58,0.13);
            padding: 1.5rem 1.7rem;
            margin-bottom: 1.5rem;
            color: #fff;
        }
        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 1rem;
            letter-spacing: 0.5px;
        }
        .section-divider {
            border: none;
            border-top: 2px solid #223344;
            margin: 1.2rem 0 1.5rem 0;
        }
        .stInfo, .stAlert, .stError {
            background-color: #223344 !important;
            color: #fff !important;
            border-radius: 0.7rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize chatbot with loading state
    if not st.session_state.initialized:
        with st.spinner("Initializing chatbot and loading models... This may take a few minutes."):
            try:
                st.session_state.chatbot = IndianStockChatbot()
                st.session_state.initialized = True
                st.success("Chatbot initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing chatbot: {str(e)}")
                st.info("The chatbot will continue with limited functionality.")

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💬 AI Chat Assistant</div>', unsafe_allow_html=True)
        st.write("Ask about Indian stocks, indices, market activity, portfolio, sentiment, or watchlist.")

        # Chat area with fixed height and scroll
        st.markdown('<div class="chat-outer">', unsafe_allow_html=True)
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for idx, (user_msg, bot_msg) in enumerate(st.session_state.history):
            st.markdown(f'''
            <div style="display: flex; flex-direction: row-reverse; align-items: flex-end;">
                <div class="chat-avatar user">🧑</div>
                <div class="chat-message user">
                    <div class="content">{user_msg}</div>
                </div>
            </div>
            <div style="display: flex; align-items: flex-end;">
                <div class="chat-avatar bot">🤖</div>
                <div class="chat-message bot{' new' if idx == len(st.session_state.history)-1 else ''}">
                    <div class="content">{bot_msg}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        # Loading animation if bot is processing
        if st.session_state.get('bot_loading', False):
            st.markdown('''
            <div class="loading-bot">
                <div class="chat-avatar bot">🤖</div>
                <div class="loading-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Quick suggestions
        st.markdown('<div style="margin-bottom: 1.2rem;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size:1.08rem; margin-bottom:0.5rem;">💡 Quick Suggestions</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, suggestion in enumerate(st.session_state.suggestions):
            if cols[i % 4].button(suggestion, key=f"suggestion_{i}", help="Click to use this suggestion"):
                st.session_state.user_input = suggestion
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Visually separate input area
        st.markdown('<hr class="input-separator" />', unsafe_allow_html=True)
        st.markdown('<div class="sticky-input">', unsafe_allow_html=True)
        # User input
        user_input = st.text_input("Type your question:", key="user_input")
        # Send button
        send_clicked = st.button("Send", key="send_button") or (user_input and st.session_state.get('last_input') != user_input)
        if send_clicked:
            if user_input.strip():
                st.session_state['bot_loading'] = True
                with st.spinner("Processing your query..."):
                    try:
                        bot_response = st.session_state.chatbot.process_query(user_input)
                        st.session_state.history.append((user_input, bot_response))
                        st.session_state.last_input = user_input
                        st.session_state['bot_loading'] = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        st.session_state.history.append((user_input, "I'm having trouble processing your request. Please try again."))
                        st.session_state['bot_loading'] = False
        st.markdown('</div>', unsafe_allow_html=True)
        # Clear chat button
        if st.button("Clear Chat", key="clear_button"):
            st.session_state.history = []
            st.session_state.last_input = ""
            st.rerun()

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Market Overview</div>', unsafe_allow_html=True)
        try:
            market_data = st.session_state.chatbot.get_market_activity()
            if market_data:
                # Nifty 50
                st.metric(
                    "Nifty 50",
                    f"₹{market_data['nifty']['current']:,.2f}",
                    f"{market_data['nifty']['change_pct']:+.2f}%"
                )
                
                # Sensex
                st.metric(
                    "Sensex",
                    f"₹{market_data['sensex']['current']:,.2f}",
                    f"{market_data['sensex']['change_pct']:+.2f}%"
                )
                
                # Market Status
                status_color = "#00e676" if market_data['market_status'] == "Open" else "#ff5252"
                st.markdown(f"Market Status: <span style='color:{status_color}; font-weight:600;'>{market_data['market_status']}</span>", unsafe_allow_html=True)
                
                # Advance-Decline Ratio
                if market_data['advance_decline']['ratio'] != float('inf'):
                    st.metric(
                        "Advance-Decline Ratio",
                        f"{market_data['advance_decline']['ratio']:.2f}",
                        f"Advances: {market_data['advance_decline']['advances']} | Declines: {market_data['advance_decline']['declines']}"
                    )
        except Exception as e:
            st.error("Unable to fetch market data")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔍 Recent Searches</div>', unsafe_allow_html=True)
        if st.session_state.history:
            recent_searches = [msg[0] for msg in st.session_state.history[-5:]]
            for i, search in enumerate(recent_searches):
                if st.button(search, key=f"recent_search_{i}_{hash(search)}"):
                    st.session_state.user_input = search
                    st.rerun()
        else:
            st.info("No recent searches")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💡 Help & Tips</div>', unsafe_allow_html=True)
        with st.expander("Show Help & Tips"):
            st.markdown("""
            **Example Queries:**
            - Stock Price: "What's the price of TCS?"
            - Market Activity: "Show market activity"
            - Sentiment: "What's the sentiment for Reliance?"
            - Watchlist: "Show my watchlist"
            - Market Terms: "Explain what is IPO?"
            - Sector Analysis: "Show sector performance"
            - Trading Signals: "What are the trading signals for HDFC Bank?"
            - Portfolio: "Show my portfolio analysis"
            
            **Tips:**
            - Be specific with company names
            - Use stock symbols for faster results
            - Ask about market terms for explanations
            - Use the quick suggestions for common queries
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Powered by AI • Real-time Market Data • Last Updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    show_chat() 
