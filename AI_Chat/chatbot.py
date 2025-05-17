import warnings
import yfinance as yf
from datetime import datetime, timedelta
import logging
from GoogleNews import GoogleNews
import re
from textblob import TextBlob
import numpy as np
from fuzzywuzzy import process
import os
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class IndianStockChatbot:
    def __init__(self):
        try:
            print("Initializing chatbot...")
            
            # Initialize prediction model
            self.prediction_model = self._initialize_prediction_model()
            
            # Initialize history
            self.history = []
            
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
            print(f"Error initializing chatbot: {str(e)}")
            self.prediction_model = None
            self.history = []
            self.stock_symbols = {}
            self.market_terms = {}
            self.intent_patterns = {}

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

    def _prepare_prediction_data(self, symbol: str) -> tuple:
        """Prepare data for prediction"""
        try:
            # Get historical data
            ticker = yf.Ticker(f"{symbol}.NS")
            hist = ticker.history(period="1y")
            
            # Calculate technical indicators
            df = pd.DataFrame(hist)
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
            df['BB_Upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
            df['BB_Lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
            
            # Prepare features
            features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal']
            data = df[features].values
            
            # Normalize data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Create sequences
            X, y = [], []
            for i in range(60, len(scaled_data)):
                X.append(scaled_data[i-60:i])
                y.append(scaled_data[i, 0])
            
            return np.array(X), np.array(y), scaler
        except Exception as e:
            logging.error(f"Error preparing prediction data: {str(e)}")
            return None, None, None

    def get_trading_signals(self, symbol: str) -> dict:
        """Generate trading signals using technical analysis and prediction"""
        try:
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            # Calculate technical indicators
            df = pd.DataFrame(hist)
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
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
            
            return signals
        except Exception as e:
            logging.error(f"Error generating trading signals: {str(e)}")
            return {}

    def clean_query(self, text):
        """Clean and normalize the query text"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def classify_intent(self, query: str) -> tuple:
        """Classify the intent of the query using pattern matching"""
        query = self.clean_query(query)
        
        # Check for exact matches in patterns
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    return intent, 1.0
        
        # If no exact match, use fuzzy matching
        best_match = None
        best_score = 0
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                score = process.fuzz.ratio(query, pattern) / 100.0
                if score > best_score:
                    best_score = score
                    best_match = intent
        
        return best_match, best_score if best_match else ('unknown', 0.0)

    def get_stock_symbol(self, user_input: str) -> str:
        """Get stock symbol from user input using fuzzy matching"""
        user_input = user_input.lower().strip()
        
        # Check for exact matches
        if user_input in self.stock_symbols:
            return self.stock_symbols[user_input]
        
        # Use fuzzy matching
        best_match = process.extractOne(user_input, self.stock_symbols.keys())
        if best_match and best_match[1] > 80:  # Threshold for fuzzy matching
            return self.stock_symbols[best_match[0]]
        
        return None

    def get_stock_details(self, symbol: str) -> dict:
        """Get detailed stock information"""
        try:
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get historical data
            hist = ticker.history(period="1mo")
            
            # Calculate basic metrics
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            # Get trading signals
            signals = self.get_trading_signals(symbol)
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'current_price': current_price,
                'change': change,
                'change_pct': change_pct,
                'volume': hist['Volume'].iloc[-1],
                'signals': signals,
                'info': info
            }
        except Exception as e:
            logging.error(f"Error getting stock details: {str(e)}")
            return {}

    def fetch_company_news(self, stock_name: str) -> list:
        """Fetch recent news articles for a company"""
        try:
            googlenews = GoogleNews(lang='en')
            googlenews.search(stock_name)
            articles = googlenews.result()
            
            # Analyze sentiment for each article
            analyzed_articles = []
            for article in articles:
                text = article['desc']
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity
                
                sentiment_label = 'positive' if sentiment > 0.1 else 'negative' if sentiment < -0.1 else 'neutral'
                
                analyzed_articles.append({
                    'title': article['title'],
                    'link': article['link'],
                    'date': article['date'],
                    'sentiment': sentiment_label,
                    'confidence': abs(sentiment)
                })
            
            return analyzed_articles
        except Exception as e:
            logging.error(f"Error fetching news: {str(e)}")
            return []

    def get_stock_analysis(self, symbol: str) -> dict:
        """Get comprehensive stock analysis"""
        try:
            details = self.get_stock_details(symbol)
            news = self.fetch_company_news(symbol)
            
            # Calculate sentiment from news
            if news:
                sentiments = [article['sentiment'] for article in news]
                positive = sentiments.count('positive')
                negative = sentiments.count('negative')
                neutral = sentiments.count('neutral')
                total = len(sentiments)
                
                sentiment_analysis = {
                    'positive_pct': (positive / total) * 100,
                    'negative_pct': (negative / total) * 100,
                    'neutral_pct': (neutral / total) * 100,
                    'overall': 'positive' if positive > negative else 'negative' if negative > positive else 'neutral'
                }
            else:
                sentiment_analysis = {
                    'positive_pct': 0,
                    'negative_pct': 0,
                    'neutral_pct': 0,
                    'overall': 'neutral'
                }
            
            return {
                'details': details,
                'news': news,
                'sentiment': sentiment_analysis
            }
        except Exception as e:
            logging.error(f"Error in stock analysis: {str(e)}")
            return {}

    def is_market_open(self) -> bool:
        """Check if the Indian stock market is currently open"""
        try:
            now = datetime.now()
            weekday = now.weekday()
            
            # Check if it's a weekday
            if weekday >= 5:  # Saturday or Sunday
                return False
            
            # Check if it's within trading hours (9:15 AM to 3:30 PM IST)
            market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            return market_start <= now <= market_end
        except Exception as e:
            logging.error(f"Error checking market status: {str(e)}")
            return False

    def process_query(self, user_input: str) -> str:
        """Process user query and generate response"""
        try:
            # Clean and classify query
            intent, confidence = self.classify_intent(user_input)
            
            # Get stock symbol if mentioned
            symbol = self.get_stock_symbol(user_input)
            
            # Process based on intent
            if intent == 'price_query' and symbol:
                details = self.get_stock_details(symbol)
                if details:
                    return f"Current price of {symbol} is ₹{details['current_price']:.2f} ({details['change_pct']:+.2f}%)"
                return f"Sorry, couldn't fetch price information for {symbol}"
            
            elif intent == 'market_status':
                is_open = self.is_market_open()
                return "Market is currently open" if is_open else "Market is currently closed"
            
            elif intent == 'index_query':
                if 'nifty' in user_input.lower():
                    return "Nifty 50 is currently at [fetch current value]"
                elif 'sensex' in user_input.lower():
                    return "Sensex is currently at [fetch current value]"
                return "Which index would you like to know about? (Nifty/Sensex)"
            
            elif intent == 'term_query':
                for term, explanation in self.market_terms.items():
                    if term in user_input.lower():
                        return explanation
                return "Which market term would you like me to explain?"
            
            elif intent == 'analysis_query' and symbol:
                analysis = self.get_stock_analysis(symbol)
                if analysis:
                    details = analysis['details']
                    sentiment = analysis['sentiment']
                    return f"""
                    Analysis for {symbol}:
                    Current Price: ₹{details['current_price']:.2f}
                    Change: {details['change_pct']:+.2f}%
                    Trading Signals: {', '.join(f'{k}: {v}' for k, v in details['signals'].items())}
                    News Sentiment: {sentiment['overall'].capitalize()} ({sentiment['positive_pct']:.1f}% positive)
                    """
                return f"Sorry, couldn't perform analysis for {symbol}"
            
            return "I'm not sure I understand. Could you please rephrase your question?"
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return "Sorry, I encountered an error while processing your request."

def main():
    chatbot = IndianStockChatbot()
    print("Chatbot initialized. Type 'quit' to exit.")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            break
        
        response = chatbot.process_query(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main() 