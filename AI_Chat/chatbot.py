import warnings
import yfinance as yf
from datetime import datetime, timedelta
import logging
from GoogleNews import GoogleNews
import re
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
import torch
import numpy as np
from fuzzywuzzy import process
import os
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import json
from textblob import TextBlob
from LSTM_Model.lstm_prediction import (
    calculate_technical_indicators as calculate_indicators_master,
    prepare_data as prepare_data_master,
    create_lstm_model as create_lstm_master,
    predict_future_prices as predict_prices_master
)


warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class IndianStockChatbot:
    def __init__(self):
        try:
            print("Initializing chatbot...")
            self._models = {} # Use a private dictionary to store loaded models

            # Initialize history
            self.history = []
            
            # Load user portfolio and watchlist
            self.portfolio = self._load_portfolio()
            self.watchlist = self._load_watchlist()
            
            # Common stock symbols mapping with aliases
            self.stock_symbols = {
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

            # Common market terms and their explanations
            self.market_terms = {
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

            # Intent patterns for better understanding
            self.intent_patterns = {
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
            
            print("Chatbot initialization completed!")
            
        except Exception as e:
            logging.error(f"Error initializing chatbot: {str(e)}")
            print(f"Error during chatbot initialization: {str(e)}")
            # Initialize with minimal functionality
            self._models = {}
            self.prediction_model = None
            self.history = []
            self.stock_symbols = {}
            self.market_terms = {}
            self.intent_patterns = {}            

    def get_model(self, model_name: str):
        """Lazy load a model and return it."""
        if model_name not in self._models:
            print(f"Loading model: {model_name}...")
            model_map = {
                'intent': "distilbert-base-uncased-finetuned-sst-2-english",
                'sentiment': "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                'text_qa': "distilbert-base-cased-distilled-squad"
            }
            if model_name in model_map:
                self._models[model_name] = pipeline(model_map[model_name])
            else:
                raise ValueError(f"Model '{model_name}' not found.")
        return self._models[model_name]

    def get_trading_signals(self, symbol: str) -> dict:
        """Generate trading signals using technical analysis and prediction"""
        try:
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            # Calculate technical indicators
            df = calculate_indicators_master(pd.DataFrame(hist))
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
            df['BB_Upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
            df['BB_Lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
            
            # Get current values
            current_price = df['Close'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            current_macd = df['MACD'].iloc[-1]
            current_macd_signal = df['MACD_Signal'].iloc[-1]
            
            # Generate signals
            signals = {
                'RSI_Signal': 'Buy' if current_rsi < 30 else 'Sell' if current_rsi > 70 else 'Neutral',
                'MACD_Signal': 'Buy' if current_macd > current_macd_signal else 'Sell',
                'BB_Signal': 'Buy' if current_price < df['BB_Lower'].iloc[-1] else 'Sell' if current_price > df['BB_Upper'].iloc[-1] else 'Neutral'
            }
            
            # Prepare prediction data
            X, y, scaler = prepare_data_master(df.copy(), sequence_length=60)
            if X.shape[0] > 0:
                n_features = X.shape[2]
                prediction_model = create_lstm_master(sequence_length=60, n_features=n_features)
                # Train model on recent data
                prediction_model.fit(X, y, epochs=10, batch_size=32, verbose=0)
                
                # Make prediction
                last_sequence = X[-1:]
                future_prices = predict_prices_master(prediction_model, last_sequence, scaler, days_to_predict=1)
                predicted_price = future_prices[0]
                
                # Calculate predicted change
                price_change = ((predicted_price - current_price) / current_price) * 100
                
                signals['Predicted_Price'] = predicted_price
                signals['Predicted_Change'] = price_change
                signals['Prediction_Signal'] = 'Buy' if price_change > 2 else 'Sell' if price_change < -2 else 'Hold'
            
            # Combine signals
            buy_signals = sum(1 for signal in signals.values() if signal == 'Buy')
            sell_signals = sum(1 for signal in signals.values() if signal == 'Sell')
            
            overall_signal = 'Buy' if buy_signals > sell_signals else 'Sell' if sell_signals > buy_signals else 'Hold'
            
            return {
                'symbol': symbol.replace('.NS', ''),
                'current_price': current_price,
                'signals': signals,
                'overall_signal': overall_signal,
                'technical_indicators': {
                    'RSI': current_rsi,
                    'MACD': current_macd,
                    'MACD_Signal': current_macd_signal,
                    'BB_Upper': df['BB_Upper'].iloc[-1],
                    'BB_Lower': df['BB_Lower'].iloc[-1]
                },
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logging.error(f"Error generating trading signals: {str(e)}")
            return None

    def clean_query(self, text):
        """Clean and normalize the query text"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s.]', '', text)
        text = ' '.join(text.split())
        return text

    def classify_intent(self, query: str) -> tuple:
        """Enhanced intent classification with confidence scores"""
        try:
            # Get base intent classification
            intent_result = self.models['intent'](query)[0]
            base_intent = intent_result['label'].lower()
            confidence = intent_result['score']
            
            # Get sentiment
            sentiment_result = self.models['sentiment'](query)[0]
            sentiment = sentiment_result['label']
            
            # Pattern-based intent refinement
            if confidence > 0.9:
                if "price" in query.lower() or "current price" in query.lower():
                    return "price_query", confidence, sentiment
                elif any(term in query.lower() for term in self.market_terms.keys()):
                    return "term_query", confidence, sentiment
                elif "news" in query.lower() or "latest" in query.lower():
                    return "news_query", confidence, sentiment
                elif "sentiment" in query.lower() or "outlook" in query.lower():
                    return "sentiment_query", confidence, sentiment
                elif "summary" in query.lower() or "summarize" in query.lower():
                    return "summary_query", confidence, sentiment
                else:
                    return base_intent, confidence, sentiment
            else:
                # Try pattern matching as fallback
                for intent, patterns in self.intent_patterns.items():
                    if any(pattern in query.lower() for pattern in patterns):
                        return intent, 0.8, sentiment
                
                return "general_query", 0.5, sentiment
                
        except Exception as e:
            logging.error(f"Error in intent classification: {str(e)}")
            return "general_query", 0.5, "neutral"

    def get_stock_symbol(self, user_input: str) -> str:
        """Enhanced stock symbol detection with dynamic lookup"""
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
            logging.error(f"Error in stock symbol detection: {str(e)}")
            return None

    def get_stock_details(self, symbol: str) -> dict:
        """Get comprehensive stock details using enhanced yfinance features"""
        try:
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            # Get financial data
            balance_sheet = ticker.balance_sheet
            income_stmt = ticker.income_stmt
            cash_flow = ticker.cashflow
            
            # Get recommendations
            recommendations = ticker.recommendations
            
            # Get earnings dates
            earnings_dates = ticker.earnings_dates
            
            # Get sustainability data
            sustainability = ticker.sustainability
            
            # Get institutional holders
            institutional_holders = ticker.institutional_holders
            
            # Get major holders
            major_holders = ticker.major_holders
            
            return {
                "company_name": info.get("longName", symbol.replace('.NS', '')),
                "current_price": info.get("currentPrice", 0),
                "day_high": info.get("dayHigh", 0),
                "day_low": info.get("dayLow", 0),
                "volume": info.get("volume", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "market_cap": info.get("marketCap", 0),
                "previous_close": info.get("previousClose", 0),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "profit_margins": info.get("profitMargins", 0),
                "operating_margins": info.get("operatingMargins", 0),
                "revenue_growth": info.get("revenueGrowth", 0),
                "earnings_growth": info.get("earningsGrowth", 0),
                "recommendation": info.get("recommendationKey", "neutral"),
                "target_price": info.get("targetMeanPrice", 0),
                "number_of_analysts": info.get("numberOfAnalystOpinions", 0),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "currency": info.get("currency", "INR"),
                "exchange": info.get("exchange", "NSE"),
                "short_name": info.get("shortName", ""),
                "long_name": info.get("longName", ""),
                "website": info.get("website", ""),
                "business_summary": info.get("longBusinessSummary", ""),
                "financial_data": {
                    "balance_sheet": balance_sheet.to_dict() if balance_sheet is not None else {},
                    "income_statement": income_stmt.to_dict() if income_stmt is not None else {},
                    "cash_flow": cash_flow.to_dict() if cash_flow is not None else {}
                },
                "recommendations": recommendations.to_dict() if recommendations is not None else {},
                "earnings_dates": earnings_dates.to_dict() if earnings_dates is not None else {},
                "sustainability": sustainability.to_dict() if sustainability is not None else {},
                "institutional_holders": institutional_holders.to_dict() if institutional_holders is not None else {},
                "major_holders": major_holders.to_dict() if major_holders is not None else {}
            }
        except Exception as e:
            logging.error(f"Error fetching stock details: {str(e)}")
            return None

    def fetch_company_news(self, stock_name: str) -> list:
        """Fetch and filter relevant company news"""
        try:
            # Initialize GoogleNews with proper settings
            googlenews = GoogleNews(lang='en', region='IN')
            googlenews.set_time_range('1d', '1d')  # Only today's news
            googlenews.search(stock_name)
            news = googlenews.result()[:5]  # Get latest 5 news items
            
            filtered_news = []
            for article in news:
                # Get full text for better analysis
                title = article['title']
                desc = article.get('desc', '')
                text = f"{title}. {desc}"
                
                # Clean and normalize the text
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Analyze sentiment with proper error handling
                try:
                    # Use a more comprehensive analysis
                    sentiment_result = self.models['sentiment'](text)[0]
                    sentiment = sentiment_result['label']
                    confidence = sentiment_result['score']
                    
                    # Additional context-based analysis
                    positive_words = ['up', 'rise', 'gain', 'positive', 'growth', 'profit', 'beat', 'surge', 'jump', 'higher']
                    negative_words = ['down', 'fall', 'loss', 'negative', 'decline', 'drop', 'miss', 'plunge', 'lower', 'worse']
                    
                    # Count positive and negative words
                    pos_count = sum(1 for word in positive_words if word in text.lower())
                    neg_count = sum(1 for word in negative_words if word in text.lower())
                    
                    # Adjust sentiment based on word counts
                    if pos_count > neg_count:
                        sentiment = 'positive'
                        confidence = max(confidence, 0.7)
                    elif neg_count > pos_count:
                        sentiment = 'negative'
                        confidence = max(confidence, 0.7)
                    else:
                        # Keep the model's sentiment if word counts are equal
                        confidence = max(confidence, 0.6)
                        
                except Exception as e:
                    logging.error(f"Error in sentiment analysis: {str(e)}")
                    sentiment = 'neutral'
                    confidence = 0.5
                
                # Get proper date
                try:
                    date = article.get('date', '')
                    if not date:
                        date = datetime.now().strftime("%Y-%m-%d %H:%M")
                    else:
                        # Convert date string to datetime
                        date_obj = datetime.strptime(date, "%m/%d/%Y")
                        date = date_obj.strftime("%Y-%m-%d %H:%M")
                except:
                    date = datetime.now().strftime("%Y-%m-%d %H:%M")
                
                filtered_news.append({
                    "title": title,
                    "link": article['link'],
                    "date": date,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "text": text
                })
            
            return filtered_news
        except Exception as e:
            logging.error(f"Error fetching company news: {str(e)}")
            return []

    def get_stock_analysis(self, symbol: str) -> dict:
        """Get comprehensive stock analysis"""
        try:
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1mo")
            
            # Calculate basic trends
            current_price = info.get("currentPrice", 0)
            prev_close = info.get("previousClose", 0)
            fifty_two_week_high = info.get("fiftyTwoWeekHigh", 0)
            fifty_two_week_low = info.get("fiftyTwoWeekLow", 0)
            
            # Calculate price change
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            # Calculate moving averages
            ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            
            return {
                "company_name": info.get("longName", symbol.replace('.NS', '')),
                "current_price": current_price,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "ma_20": ma_20,
                "ma_50": ma_50,
                "fifty_two_week_high": fifty_two_week_high,
                "fifty_two_week_low": fifty_two_week_low,
                "volume": info.get("volume", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "market_cap": info.get("marketCap", 0),
                "recommendation": info.get("recommendationKey", "neutral")
            }
        except Exception as e:
            logging.error(f"Error fetching stock analysis: {str(e)}")
            return None

    def get_market_activity(self) -> dict:
        """Get overall market activity and indices using enhanced yfinance features"""
        try:
            # Get Nifty 50 data with real-time updates
            nifty = yf.Ticker("^NSEI")
            nifty_data = nifty.history(period="1d", interval="1m")  # Changed to 1-minute intervals
            nifty_info = nifty.info
            
            # Get Sensex data with real-time updates
            sensex = yf.Ticker("^BSESN")
            sensex_data = sensex.history(period="1d", interval="1m")  # Changed to 1-minute intervals
            sensex_info = sensex.info
            
            # Get market status with real-time check
            market_status = "Open" if self.is_market_open() else "Closed"
            
            # Get sector performance with real-time data
            sectors = {
                "IT": "^CNXIT",
                "Bank": "^NSEBANK",
                "Pharma": "^CNXPHARMA",
                "Auto": "^CNXAUTO",
                "FMCG": "^CNXFMCG"
            }
            
            sector_performance = {}
            for sector_name, sector_symbol in sectors.items():
                try:
                    sector_ticker = yf.Ticker(sector_symbol)
                    sector_data = sector_ticker.history(period="1d", interval="1m")  # Changed to 1-minute intervals
                    if not sector_data.empty:
                        change_pct = ((sector_data['Close'].iloc[-1] - sector_data['Open'].iloc[0]) / 
                                    sector_data['Open'].iloc[0]) * 100
                        sector_performance[sector_name] = {
                            "current": sector_data['Close'].iloc[-1],
                            "change_pct": change_pct,
                            "last_update": sector_data.index[-1].strftime("%H:%M:%S")
                        }
                except Exception as e:
                    logging.error(f"Error fetching {sector_name} sector data: {str(e)}")
            
            # Calculate changes using latest data
            nifty_change = nifty_data['Close'].iloc[-1] - nifty_data['Open'].iloc[0]
            nifty_change_pct = (nifty_change / nifty_data['Open'].iloc[0]) * 100
            
            sensex_change = sensex_data['Close'].iloc[-1] - sensex_data['Open'].iloc[0]
            sensex_change_pct = (sensex_change / sensex_data['Open'].iloc[0]) * 100
            
            # Get advance-decline ratio
            advance_decline = self.get_advance_decline_ratio()
            
            return {
                "nifty": {
                    "current": nifty_data['Close'].iloc[-1],
                    "change": nifty_change,
                    "change_pct": nifty_change_pct,
                    "high": nifty_data['High'].iloc[-1],
                    "low": nifty_data['Low'].iloc[-1],
                    "volume": nifty_data['Volume'].iloc[-1],
                    "open": nifty_data['Open'].iloc[0],
                    "prev_close": nifty_info.get("previousClose", 0),
                    "last_update": nifty_data.index[-1].strftime("%H:%M:%S")
                },
                "sensex": {
                    "current": sensex_data['Close'].iloc[-1],
                    "change": sensex_change,
                    "change_pct": sensex_change_pct,
                    "high": sensex_data['High'].iloc[-1],
                    "low": sensex_data['Low'].iloc[-1],
                    "volume": sensex_data['Volume'].iloc[-1],
                    "open": sensex_data['Open'].iloc[0],
                    "prev_close": sensex_info.get("previousClose", 0),
                    "last_update": sensex_data.index[-1].strftime("%H:%M:%S")
                },
                "market_status": market_status,
                "advance_decline": advance_decline,
                "sector_performance": sector_performance,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logging.error(f"Error fetching market activity: {str(e)}")
            return None

    def get_index_data(self, index_symbol: str) -> dict:
        """Get detailed data for a specific index"""
        try:
            if index_symbol.upper() == "NIFTY":
                ticker = yf.Ticker("^NSEI")
            elif index_symbol.upper() == "SENSEX":
                ticker = yf.Ticker("^BSESN")
            else:
                return None
            
            # Get historical data
            hist = ticker.history(period="1d")
            
            # Calculate changes
            current = hist['Close'].iloc[-1]
            open_price = hist['Open'].iloc[0]
            high = hist['High'].iloc[-1]
            low = hist['Low'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            
            change = current - open_price
            change_pct = (change / open_price) * 100
            
            return {
                "symbol": index_symbol.upper(),
                "current": current,
                "change": change,
                "change_pct": change_pct,
                "open": open_price,
                "high": high,
                "low": low,
                "volume": volume,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logging.error(f"Error fetching index data: {str(e)}")
            return None

    def get_portfolio_analysis(self, symbols: list) -> dict:
        """Analyze a portfolio of stocks"""
        try:
            portfolio = {}
            total_value = 0
            total_change = 0
            
            for symbol in symbols:
                if not symbol.endswith('.NS'):
                    symbol = f"{symbol}.NS"
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                current_price = info.get("currentPrice", 0)
                prev_close = info.get("previousClose", 0)
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                
                portfolio[symbol.replace('.NS', '')] = {
                    "price": current_price,
                    "change": change,
                    "change_pct": change_pct,
                    "pe_ratio": info.get("trailingPE", 0),
                    "market_cap": info.get("marketCap", 0)
                }
                
                total_value += current_price
                total_change += change
            
            return {
                "stocks": portfolio,
                "total_value": total_value,
                "total_change": total_change,
                "total_change_pct": (total_change / total_value) * 100 if total_value > 0 else 0
            }
        except Exception as e:
            logging.error(f"Error analyzing portfolio: {str(e)}")
            return None

    def get_sentiment_analysis(self, symbol: str) -> dict:
        """Get detailed sentiment analysis for a stock using enhanced yfinance features"""
        try:
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            # Get stock data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            info = ticker.info
            
            if hist.empty:
                return {
                    "symbol": symbol.replace('.NS', ''),
                    "error": "No price data available"
                }
            
            # Get analyst recommendations
            recommendations = ticker.recommendations
            recommendation_summary = ticker.recommendations_summary
            
            # Get earnings estimates
            earnings_estimate = ticker.earnings_estimate
            revenue_estimate = ticker.revenue_estimate
            
            # Get earnings history
            earnings_history = ticker.earnings_history
            
            # Get EPS trend
            eps_trend = ticker.eps_trend
            
            # Get EPS revisions
            eps_revisions = ticker.eps_revisions
            
            # Get growth estimates
            growth_estimates = ticker.growth_estimates
            
            # Calculate price change
            current_price = hist['Close'].iloc[-1]
            open_price = hist['Open'].iloc[0]
            price_change = current_price - open_price
            price_change_pct = (price_change / open_price) * 100
            
            # Get news and analyze sentiment
            news = self.fetch_company_news(symbol)
            
            if not news:
                return {
                    "symbol": symbol.replace('.NS', ''),
                    "sentiment_score": 0,
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "total_news": 0,
                    "recent_news": [],
                    "price_change": price_change_pct,
                    "current_price": current_price,
                    "message": "No recent news found for analysis."
                }
            
            # Calculate sentiment scores
            positive_news = [article for article in news if article['sentiment'] == 'positive']
            negative_news = [article for article in news if article['sentiment'] == 'negative']
            neutral_news = [article for article in news if article['sentiment'] == 'neutral']
            
            positive_count = len(positive_news)
            negative_count = len(negative_news)
            neutral_count = len(neutral_news)
            
            # Calculate weighted scores
            positive_score = sum(article['confidence'] for article in positive_news)
            negative_score = sum(article['confidence'] for article in negative_news)
            neutral_score = sum(article['confidence'] for article in neutral_news)
            
            # Calculate overall sentiment score (-100 to +100)
            total_score = positive_score + negative_score + neutral_score
            if total_score > 0:
                sentiment_score = ((positive_score - negative_score) / total_score) * 100
            else:
                sentiment_score = 0
            
            # Determine market context
            market_context = "bullish" if price_change_pct > 0 else "bearish"
            
            # Sort news by date (most recent first)
            news.sort(key=lambda x: x['date'], reverse=True)
            
            return {
                "symbol": symbol.replace('.NS', ''),
                "sentiment_score": sentiment_score,
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
                "total_news": len(news),
                "recent_news": news[:3],
                "price_change": price_change_pct,
                "current_price": current_price,
                "market_context": market_context,
                "analyst_recommendations": recommendations.to_dict() if recommendations is not None else {},
                "recommendation_summary": recommendation_summary.to_dict() if recommendation_summary is not None else {},
                "earnings_estimate": earnings_estimate.to_dict() if earnings_estimate is not None else {},
                "revenue_estimate": revenue_estimate.to_dict() if revenue_estimate is not None else {},
                "earnings_history": earnings_history.to_dict() if earnings_history is not None else {},
                "eps_trend": eps_trend.to_dict() if eps_trend is not None else {},
                "eps_revisions": eps_revisions.to_dict() if eps_revisions is not None else {},
                "growth_estimates": growth_estimates.to_dict() if growth_estimates is not None else {},
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {str(e)}")
            return None

    def get_watchlist_analysis(self, symbols: list) -> dict:
        """Analyze stocks in watchlist"""
        try:
            watchlist = {}
            alerts = []
            
            for symbol in symbols:
                if not symbol.endswith('.NS'):
                    symbol = f"{symbol}.NS"
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                current_price = info.get("currentPrice", 0)
                prev_close = info.get("previousClose", 0)
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                
                watchlist[symbol.replace('.NS', '')] = {
                    "price": current_price,
                    "change": change,
                    "change_pct": change_pct,
                    "volume": info.get("volume", 0),
                    "pe_ratio": info.get("trailingPE", 0)
                }
                
                # Generate alerts
                if abs(change_pct) > 5:  # Alert for >5% price change
                    alerts.append(f"{symbol.replace('.NS', '')} has moved {change_pct:+.2f}%")
                
                if info.get("volume", 0) > info.get("averageVolume", 0) * 2:  # Alert for high volume
                    alerts.append(f"{symbol.replace('.NS', '')} has unusually high volume")
            
            return {
                "stocks": watchlist,
                "alerts": alerts,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logging.error(f"Error analyzing watchlist: {str(e)}")
            return None

    def get_advance_decline_ratio(self) -> dict:
        """Get advance-decline ratio for the market"""
        try:
            nifty = yf.Ticker("^NSEI")
            components = nifty.info.get("components", [])
            
            advances = 0
            declines = 0
            
            for symbol in components[:50]:  # Check top 50 stocks
                if not symbol.endswith('.NS'):
                    symbol = f"{symbol}.NS"
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info.get("currentPrice", 0) > info.get("previousClose", 0):
                    advances += 1
                else:
                    declines += 1
            
            return {
                "advances": advances,
                "declines": declines,
                "ratio": advances / declines if declines > 0 else float('inf')
            }
        except Exception as e:
            logging.error(f"Error calculating advance-decline ratio: {str(e)}")
            return {"advances": 0, "declines": 0, "ratio": 0}

    def get_sector_analysis(self, sector_key: str) -> dict:
        """Get detailed analysis for a specific sector"""
        try:
            sector = yf.Sector(sector_key)
            
            # Get sector overview
            overview = sector.overview
            
            # Get top companies
            top_companies = sector.top_companies
            
            # Get top ETFs
            top_etfs = sector.top_etfs
            
            # Get top mutual funds
            top_mutual_funds = sector.top_mutual_funds
            
            # Get industries in the sector
            industries = sector.industries
            
            # Get research reports
            research_reports = sector.research_reports
            
            return {
                "name": sector.name,
                "key": sector.key,
                "overview": overview,
                "top_companies": top_companies.to_dict() if top_companies is not None else {},
                "top_etfs": top_etfs,
                "top_mutual_funds": top_mutual_funds,
                "industries": industries.to_dict() if industries is not None else {},
                "research_reports": research_reports,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logging.error(f"Error in sector analysis: {str(e)}")
            return None

    def get_industry_analysis(self, industry_key: str) -> dict:
        """Get detailed analysis for a specific industry"""
        try:
            industry = yf.Industry(industry_key)
            
            # Get industry overview
            overview = industry.overview
            
            # Get top companies
            top_companies = industry.top_companies
            
            # Get top performing companies
            top_performing = industry.top_performing_companies
            
            # Get top growth companies
            top_growth = industry.top_growth_companies
            
            # Get research reports
            research_reports = industry.research_reports
            
            return {
                "name": industry.name,
                "key": industry.key,
                "sector_key": industry.sector_key,
                "sector_name": industry.sector_name,
                "overview": overview,
                "top_companies": top_companies.to_dict() if top_companies is not None else {},
                "top_performing_companies": top_performing.to_dict() if top_performing is not None else {},
                "top_growth_companies": top_growth.to_dict() if top_growth is not None else {},
                "research_reports": research_reports,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logging.error(f"Error in industry analysis: {str(e)}")
            return None

    def generate_detailed_response(self, intent: str, data: dict, sentiment: str) -> str:
        """Generate detailed and natural responses with enhanced data"""
        try:
            # Add sector analysis response
            if intent == 'sector_analysis' and data:
                response = (
                    f"Sector Analysis for {data['name']}:\n"
                    f"Overview: {data['overview']}\n\n"
                )
                
                # Add top companies
                if data.get('top_companies'):
                    response += "Top Companies:\n"
                    companies = data['top_companies']
                    for company in companies[:5]:  # Show top 5 companies
                        response += f"• {company['name']} ({company['symbol']})\n"
                
                # Add top ETFs
                if data.get('top_etfs'):
                    response += "\nTop ETFs:\n"
                    for symbol, name in data['top_etfs'].items():
                        response += f"• {name} ({symbol})\n"
                
                # Add industries
                if data.get('industries'):
                    response += "\nIndustries in this Sector:\n"
                    industries = data['industries']
                    for industry in industries[:5]:  # Show top 5 industries
                        response += f"• {industry['name']} ({industry['market_weight']:.2f}%)\n"
                
                response += f"\nLast Updated: {data['last_updated']}"
                return response
            
            # Add industry analysis response
            elif intent == 'industry_analysis' and data:
                response = (
                    f"Industry Analysis for {data['name']}:\n"
                    f"Sector: {data['sector_name']}\n"
                    f"Overview: {data['overview']}\n\n"
                )
                
                # Add top companies
                if data.get('top_companies'):
                    response += "Top Companies:\n"
                    companies = data['top_companies']
                    for company in companies[:5]:  # Show top 5 companies
                        response += f"• {company['name']} ({company['symbol']})\n"
                
                # Add top performing companies
                if data.get('top_performing_companies'):
                    response += "\nTop Performing Companies:\n"
                    companies = data['top_performing_companies']
                    for company in companies[:3]:  # Show top 3 performers
                        response += f"• {company['name']} ({company['symbol']}) - {company['performance']:.2f}%\n"
                
                # Add top growth companies
                if data.get('top_growth_companies'):
                    response += "\nTop Growth Companies:\n"
                    companies = data['top_growth_companies']
                    for company in companies[:3]:  # Show top 3 growth companies
                        response += f"• {company['name']} ({company['symbol']}) - {company['growth']:.2f}%\n"
                
                response += f"\nLast Updated: {data['last_updated']}"
                return response
            
            if intent == 'price_query' and data:
                response = (
                    f"The current price of {data['company_name']} is ₹{data['current_price']:.2f}. "
                    f"Today's high is ₹{data['day_high']:.2f} and low is ₹{data['day_low']:.2f}. "
                    f"Volume traded: {data['volume']:,}, P/E ratio: {data['pe_ratio']:.2f}, "
                    f"Market Cap: ₹{data['market_cap']/10000000:.2f} Cr.\n\n"
                )
                
                # Add financial metrics
                if data.get('financial_data'):
                    response += "Key Financial Metrics:\n"
                    if data['financial_data'].get('income_statement'):
                        income = data['financial_data']['income_statement']
                        if income:
                            response += f"- Revenue Growth: {data.get('revenue_growth', 0)*100:.2f}%\n"
                            response += f"- Profit Margins: {data.get('profit_margins', 0)*100:.2f}%\n"
                            response += f"- Operating Margins: {data.get('operating_margins', 0)*100:.2f}%\n"
                
                # Add analyst recommendations
                if data.get('recommendations'):
                    response += "\nAnalyst Recommendations:\n"
                    recs = data['recommendations']
                    if recs:
                        response += f"- Consensus: {data.get('recommendation', 'neutral').upper()}\n"
                        response += f"- Target Price: ₹{data.get('target_price', 0):.2f}\n"
                        response += f"- Number of Analysts: {data.get('number_of_analysts', 0)}\n"
                
                # Add business summary
                if data.get('business_summary'):
                    response += f"\nBusiness Summary:\n{data['business_summary'][:200]}...\n"
                
                return response
            
            elif intent == 'market_activity':
                if data:
                    # Build response without market status
                    response = []
                    response.append(f"Nifty 50: ₹{data['nifty']['current']:.2f} ({data['nifty']['change_pct']:+.2f}%)")
                    response.append(f"High: ₹{data['nifty']['high']:.2f}, Low: ₹{data['nifty']['low']:.2f}")
                    response.append(f"Sensex: ₹{data['sensex']['current']:.2f} ({data['sensex']['change_pct']:+.2f}%)")
                    response.append(f"High: ₹{data['sensex']['high']:.2f}, Low: ₹{data['sensex']['low']:.2f}")
                    
                    # Add sector performance
                    if data.get('sector_performance'):
                        response.append("\n\nSector Performance:")
                        for sector, perf in data['sector_performance'].items():
                            response.append(f"\n{sector}: {perf['change_pct']:+.2f}%")
                    
                    if data['advance_decline']['ratio'] != float('inf'):
                        response.append(f"\n\nAdvance-Decline Ratio: {data['advance_decline']['ratio']:.2f}")
                    
                    response.append(f"\n\nLast Updated: {data['last_updated']}")
                    return "\n".join(response)
            
            elif intent == 'sentiment_analysis' and data:
                if 'error' in data:
                    return f"Error: {data['error']}"
                
                response = (
                    f"Sentiment Analysis for {data['symbol']}:\n"
                    f"Current Price: ₹{data['current_price']:.2f} ({data['price_change']:+.2f}%)\n"
                    f"Overall Sentiment Score: {data['sentiment_score']:.2f}\n"
                    f"Market Context: {data['market_context']}\n\n"
                    f"News Analysis:\n"
                    f"- Positive News: {data['positive']}\n"
                    f"- Negative News: {data['negative']}\n"
                    f"- Neutral News: {data['neutral']}\n"
                    f"Total News Analyzed: {data['total_news']}\n"
                )
                
                # Add analyst recommendations
                if data.get('analyst_recommendations'):
                    response += "\nAnalyst Recommendations:\n"
                    if data.get('recommendation_summary'):
                        summary = data['recommendation_summary']
                        response += f"- Consensus: {summary.get('recommendation', 'neutral').upper()}\n"
                        response += f"- Target Price: ₹{summary.get('targetMeanPrice', 0):.2f}\n"
                
                # Add earnings estimates
                if data.get('earnings_estimate'):
                    response += "\nEarnings Estimates:\n"
                    earnings = data['earnings_estimate']
                    if earnings:
                        response += f"- Next Quarter: ₹{earnings.get('nextQuarter', 0):.2f}\n"
                        response += f"- Current Year: ₹{earnings.get('currentYear', 0):.2f}\n"
                
                # Add growth estimates
                if data.get('growth_estimates'):
                    response += "\nGrowth Estimates:\n"
                    growth = data['growth_estimates']
                    if growth:
                        response += f"- Next 5 Years: {growth.get('next5Years', 0)*100:.2f}%\n"
                
                # Add recent news
                response += "\nRecent News:\n"
                for article in data['recent_news']:
                    response += (
                        f"• {article['title']}\n"
                        f"  Sentiment: {article['sentiment'].upper()} "
                        f"(Confidence: {article['confidence']:.2f})\n"
                        f"  Time: {article['date']}\n"
                    )
                
                response += f"\nLast Updated: {data['last_updated']}"
                return response
            
            elif intent == 'analysis_query' and data:
                analysis = self.get_stock_analysis(data.get('symbol'))
                if analysis:
                    trend = "upward" if analysis['price_change'] > 0 else "downward"
                    ma_trend = "above" if analysis['current_price'] > analysis['ma_20'] else "below"
                    
                    response = (
                        f"{analysis['company_name']} is currently trading at ₹{analysis['current_price']:.2f}, "
                        f"showing a {trend} trend with {abs(analysis['price_change_pct']):.2f}% change. "
                        f"The stock is trading {ma_trend} its 20-day moving average. "
                        f"Market cap: ₹{analysis['market_cap']/10000000:.2f} Cr, "
                        f"P/E ratio: {analysis['pe_ratio']:.2f}. "
                        f"Analyst recommendation: {analysis['recommendation']}.\n\n"
                    )
                    
                    # Add financial analysis
                    if data.get('financial_data'):
                        response += "Financial Analysis:\n"
                        financial = data['financial_data']
                        if financial.get('income_statement'):
                            response += f"- Revenue Growth: {data.get('revenue_growth', 0)*100:.2f}%\n"
                            response += f"- Profit Margins: {data.get('profit_margins', 0)*100:.2f}%\n"
                            response += f"- Operating Margins: {data.get('operating_margins', 0)*100:.2f}%\n"
                    
                    # Add technical analysis
                    response += "\nTechnical Analysis:\n"
                    response += f"- 20-day MA: ₹{analysis['ma_20']:.2f}\n"
                    response += f"- 50-day MA: ₹{analysis['ma_50']:.2f}\n"
                    response += f"- 52-week High: ₹{analysis['fifty_two_week_high']:.2f}\n"
                    response += f"- 52-week Low: ₹{analysis['fifty_two_week_low']:.2f}\n"
                    
                    # Add news context
                    news = self.fetch_company_news(analysis['company_name'])
                    if news:
                        response += "\nRecent Developments:\n"
                        for article in news[:2]:
                            response += f"• {article['title']} ({article['sentiment'].upper()})\n"
                    
                    return response
            
            return "I can help you with stock prices, news, and market terms. What would you like to know?"
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "I'm having trouble understanding. Could you please rephrase your question?"

    def process_query(self, user_input: str) -> str:
        """Process user query with enhanced functionality"""
        try:
            # Add to history
            self.history.append(user_input)
            
            # Clean and normalize the query
            cleaned_query = self.clean_query(user_input)
            
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
            
            # Check for sentiment analysis queries
            if 'sentiment' in cleaned_query.lower():
                symbol = self.get_stock_symbol(cleaned_query)
                if symbol:
                    data = self.get_sentiment_analysis(symbol)
                    if data:
                        if 'error' in data:
                            return f"Error: {data['error']}"
                        
                        response = (
                            f"Sentiment Analysis for {data['symbol']}:\n"
                            f"Current Price: ₹{data['current_price']:.2f} ({data['price_change']:+.2f}%)\n"
                            f"Overall Sentiment Score: {data['sentiment_score']:.2f}\n"
                            f"Market Context: {data['market_context']}\n\n"
                            f"News Analysis:\n"
                            f"- Positive News: {data['positive']}\n"
                            f"- Negative News: {data['negative']}\n"
                            f"- Neutral News: {data['neutral']}\n"
                            f"Total News Analyzed: {data['total_news']}\n"
                        )
                        
                        if data.get('recent_news'):
                            response += "\nRecent News:\n"
                            for article in data['recent_news']:
                                response += (
                                    f"• {article['title']}\n"
                                    f"  Sentiment: {article['sentiment'].upper()} "
                                    f"(Confidence: {article['confidence']:.2f})\n"
                                    f"  Time: {article['date']}\n"
                                )
                        
                        response += f"\nLast Updated: {data['last_updated']}"
                        return response
                    return f"Unable to fetch sentiment analysis for {symbol} at the moment."
                return "Please specify which stock's sentiment you'd like to know about."
            
            # Check for sector performance queries
            if any(word in cleaned_query.lower() for word in ['sector performance', 'sector', 'sectors']):
                data = self.get_market_activity()
                if data and 'sector_performance' in data:
                    response = "Sector Performance:\n"
                    for sector, perf in data['sector_performance'].items():
                        response += f"{sector}: {perf['change_pct']:+.2f}%\n"
                    return response
                return "Unable to fetch sector performance data at the moment."
            
            # Check for IPO queries
            if 'ipo' in cleaned_query.lower():
                return "IPO (Initial Public Offering) is when a private company offers its shares to the public for the first time. It allows companies to raise capital from public investors and provides liquidity to existing shareholders."
            
            # Check for trading signal queries
            if any(phrase in cleaned_query.lower() for phrase in ['trading signal', 'buy signal', 'sell signal', 'when to buy', 'when to sell']):
                symbol = self.get_stock_symbol(cleaned_query)
                if symbol:
                    data = self.get_trading_signals(symbol)
                    if data:
                        response = f"Trading Signals for {data['symbol']}:\n"
                        response += f"Current Price: ₹{data['current_price']:.2f}\n"
                        response += f"RSI Signal: {data['signals']['RSI_Signal']}\n"
                        response += f"MACD Signal: {data['signals']['MACD_Signal']}\n"
                        response += f"Bollinger Bands Signal: {data['signals']['BB_Signal']}\n"
                        if 'Prediction_Signal' in data['signals']:
                            response += f"Prediction Signal: {data['signals']['Prediction_Signal']}\n"
                        response += f"Overall Signal: {data['overall_signal']}"
                        return response
                return "Please specify which stock's trading signals you'd like to know about."
            
            # Check for portfolio analysis queries
            if any(word in cleaned_query.lower() for word in ['portfolio', 'my stocks', 'my investments']):
                portfolio_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']  # You can make this dynamic
                data = self.get_portfolio_analysis(portfolio_symbols)
                if data:
                    response = "Portfolio Analysis:\n"
                    for symbol, details in data['stocks'].items():
                        response += f"\n{symbol}:\n"
                        response += f"Price: ₹{details['price']:.2f}\n"
                        response += f"Change: {details['change_pct']:+.2f}%\n"
                        response += f"P/E Ratio: {details['pe_ratio']:.2f}\n"
                    response += f"\nTotal Value: ₹{data['total_value']:.2f}"
                    response += f"\nTotal Change: {data['total_change_pct']:+.2f}%"
                    return response
                return "Unable to fetch portfolio analysis at the moment."
            
            # Check for watchlist queries
            if any(word in cleaned_query.lower() for word in ['watchlist', 'watch list']):
                watchlist_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']  # You can make this dynamic
                data = self.get_watchlist_analysis(watchlist_symbols)
                if data:
                    response = "Watchlist Analysis:\n"
                    for symbol, details in data['stocks'].items():
                        response += f"\n{symbol}:\n"
                        response += f"Price: ₹{details['price']:.2f}\n"
                        response += f"Change: {details['change_pct']:+.2f}%\n"
                        response += f"Volume: {details['volume']:,}\n"
                        response += f"P/E Ratio: {details['pe_ratio']:.2f}\n"
                    if data['alerts']:
                        response += "\nAlerts:\n"
                        for alert in data['alerts']:
                            response += f"- {alert}\n"
                    return response
                return "Unable to fetch watchlist analysis at the moment."
            
            # Get intent and sentiment
            intent, confidence, sentiment = self.classify_intent(cleaned_query)
            
            # Extract stock symbol
            symbol = self.get_stock_symbol(cleaned_query)
            
            # Get relevant data based on intent
            data = None
            if intent == 'price_query' and symbol:
                data = self.get_stock_details(symbol)
            elif intent == 'news_query' and symbol:
                data = self.fetch_company_news(symbol)
            elif intent == 'term_query':
                for term, explanation in self.market_terms.items():
                    if term in cleaned_query:
                        data = {'term': term, 'explanation': explanation}
                        break
            elif intent in ['analysis_query', 'summary_query'] and symbol:
                data = {'symbol': symbol}
            
            # Generate response
            if not symbol and intent in ['price_query', 'news_query', 'analysis_query', 'summary_query', 'sentiment_analysis']:
                return "Could you please specify the full company name or stock symbol?"
            
            response = self.generate_detailed_response(intent, data, sentiment)
            return response
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return "I'm having trouble understanding. Could you please rephrase your question?"

    def _load_portfolio(self) -> dict:
        """Load user portfolio from file"""
        try:
            if os.path.exists('portfolio.json'):
                with open('portfolio.json', 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Error loading portfolio: {str(e)}")
            return {}

    def _load_watchlist(self) -> dict:
        """Load user watchlist from file"""
        try:
            if os.path.exists('watchlist.json'):
                with open('watchlist.json', 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Error loading watchlist: {str(e)}")
            return {}

    def add_to_portfolio(self, symbol: str, quantity: int) -> bool:
        """Add stock to portfolio"""
        try:
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            if symbol in self.portfolio:
                self.portfolio[symbol]['quantity'] += quantity
            else:
                self.portfolio[symbol] = {'quantity': quantity}
            
            with open('portfolio.json', 'w') as f:
                json.dump(self.portfolio, f)
            return True
        except Exception as e:
            logging.error(f"Error adding to portfolio: {str(e)}")
            return False

    def add_to_watchlist(self, symbol: str) -> bool:
        """Add stock to watchlist"""
        try:
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            if symbol not in self.watchlist:
                self.watchlist[symbol] = {'added_date': datetime.now().strftime("%Y-%m-%d")}
                with open('watchlist.json', 'w') as f:
                    json.dump(self.watchlist, f)
                return True
            return False
        except Exception as e:
            logging.error(f"Error adding to watchlist: {str(e)}")
            return False

    def is_market_open(self) -> bool:
        """Check if Indian market is currently open with real-time data"""
        try:
            # Get the current time in India's timezone (IST is UTC+5:30)
            now_utc = datetime.utcnow()
            now_ist = now_utc + timedelta(hours=5, minutes=30)

            # Check if it's a weekday and within market hours
            if now_ist.weekday() >= 5:  # Saturday or Sunday
                return False

            market_open_time = now_ist.replace(hour=9, minute=15, second=0, microsecond=0).time()
            market_close_time = now_ist.replace(hour=15, minute=30, second=0, microsecond=0).time()

            if not (market_open_time <= now_ist.time() <= market_close_time):
                return False

            # Check for recent trading activity on Nifty 50 as a proxy for market being open
            nifty = yf.Ticker("^NSEI")
            # Fetch data for the last 5 minutes. If it's empty, the market is likely closed (e.g., holiday).
            hist = nifty.history(period="5m", interval="1m")
            return not hist.empty

        except Exception as e:
            logging.error(f"Error checking market status: {str(e)}")
            # Fallback to time-based check if there's an error
            try:
                now = datetime.now()
                if now.weekday() >= 5:  # Weekend
                    return False
                market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
                return market_open <= now <= market_close
            except:
                return False

def main():
    try:
        print("Initializing chatbot...")
        chatbot = IndianStockChatbot()
        print("Chatbot initialized successfully!")
        print("\nYou can now ask questions about Indian stocks, market activity, or market terms.")
        print("Type 'quit' or 'exit' to end the conversation.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                response = chatbot.process_query(user_input)
                print(f"Bot: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                print("I'm having trouble processing your request. Please try again.")
                
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        print("Error: Failed to initialize the chatbot. Please check the logs for details.")
        print("Try running the script again.")

if __name__ == "__main__":
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main() 
