import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import json
import os
import sys
from textblob import TextBlob

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from chatbot import IndianStockChatbot as MLChatbot  # Import using absolute path

class IndianStockChatbot:
    def __init__(self):
        """Initialize the chatbot with ML capabilities"""
        self.ml_chatbot = MLChatbot()  # Initialize the ML-powered chatbot
        self.market_status = self.get_market_status()
        self.watchlist = self.load_watchlist()
        self.portfolio = self.load_portfolio()
    
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
            # Fallback to time-based check if there's an error
            try:
                now = datetime.now()
                market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
                return "Open" if market_open <= now <= market_close else "Closed"
            except:
                return "Unknown"
    
    def load_watchlist(self):
        """Load watchlist data"""
        try:
            if os.path.exists("Watchlist/watchlist.json"):
                with open("Watchlist/watchlist.json", "r") as f:
                    return json.load(f)
            return []
        except Exception as e:
            return []
    
    def load_portfolio(self):
        """Load portfolio data"""
        try:
            if os.path.exists("Portfolio/portfolio_data.json"):
                with open("Portfolio/portfolio_data.json", "r") as f:
                    return json.load(f)
            return {"holdings": {}, "transactions": []}
        except Exception as e:
            return {"holdings": {}, "transactions": []}
    
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
            return None
    
    def get_market_activity(self):
        """Get real-time market activity data"""
        try:
            # Get Nifty 50 data with real-time updates
            nifty = yf.Ticker("^NSEI")
            nifty_data = nifty.history(period="1d", interval="1m")  # Use 1-minute intervals for real-time data
            
            # Get Sensex data with real-time updates
            sensex = yf.Ticker("^BSESN")
            sensex_data = sensex.history(period="1d", interval="1m")  # Use 1-minute intervals for real-time data
            
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
            return {'advances': 0, 'declines': 0, 'unchanged': 0, 'ratio': 0}
    
    def get_sentiment(self, symbol):
        """Get sentiment analysis for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return None
            
            sentiments = []
            for article in news[:10]:
                if 'title' in article:
                    blob = TextBlob(article['title'])
                    sentiments.append(blob.sentiment.polarity)
            
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                return {
                    'average': avg_sentiment,
                    'label': "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
                }
            return None
        except Exception as e:
            return None
    
    def process_query(self, query):
        """Process user query using ML models and generate response"""
        try:
            # First try to get a response from the ML chatbot
            try:
                ml_response = self.ml_chatbot.process_query(query)
                if ml_response and ml_response != "I'm having trouble understanding. Could you please rephrase your question?":
                    return ml_response
            except Exception as ml_error:
                st.error(f"ML processing error: {str(ml_error)}")
            
            # If ML chatbot fails or returns a generic response, try specific query handling
            query = query.lower()
            
            # Check for stock price query
            if any(word in query for word in ["price", "value", "current price", "stock price"]):
                # Extract stock symbol using ML chatbot's symbol detection
                try:
                    symbol = self.ml_chatbot.get_stock_symbol(query)
                    if symbol:
                        info = self.get_stock_info(symbol)
                        if info:
                            return f"The current price of {info['name']} ({info['symbol']}) is ₹{info['price']:,.2f} ({info['change_percent']:+.2f}%)"
                except:
                    # Fallback to basic symbol detection
                    words = query.split()
                    for word in words:
                        if word.upper() in ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]:
                            info = self.get_stock_info(word)
                            if info:
                                return f"The current price of {info['name']} ({info['symbol']}) is ₹{info['price']:,.2f} ({info['change_percent']:+.2f}%)"
            
            # Check for market activity query
            if "market" in query and ("activity" in query or "status" in query):
                activity = self.get_market_activity()
                if activity:
                    return f"Nifty 50: ₹{activity['nifty']['current']:,.2f} ({activity['nifty']['change_pct']:+.2f}%)\nSensex: ₹{activity['sensex']['current']:,.2f} ({activity['sensex']['change_pct']:+.2f}%)"
            
            # Check for sentiment query
            if "sentiment" in query:
                try:
                    # Use ML chatbot's sentiment analysis
                    symbol = self.ml_chatbot.get_stock_symbol(query)
                    if symbol:
                        sentiment_data = self.ml_chatbot.get_sentiment_analysis(symbol)
                        if sentiment_data:
                            return (
                                f"Sentiment Analysis for {symbol}:\n"
                                f"Overall Sentiment Score: {sentiment_data['sentiment_score']:.2f}\n"
                                f"Market Context: {sentiment_data['market_context']}\n"
                                f"Positive News: {sentiment_data['positive']}\n"
                                f"Negative News: {sentiment_data['negative']}\n"
                                f"Neutral News: {sentiment_data['neutral']}\n"
                                f"Total News Analyzed: {sentiment_data['total_news']}"
                            )
                except:
                    # Fallback to basic sentiment analysis
                    words = query.split()
                    for word in words:
                        if word.upper() in ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]:
                            sentiment = self.get_sentiment(f"{word}.NS")
                            if sentiment:
                                return f"The sentiment for {word} is {sentiment['label']} (Score: {sentiment['average']:.2f})"
            
            # Check for trading signals query
            if any(phrase in query for phrase in ["trading signal", "buy signal", "sell signal", "when to buy", "when to sell"]):
                try:
                    symbol = self.ml_chatbot.get_stock_symbol(query)
                    if symbol:
                        signals = self.ml_chatbot.get_trading_signals(symbol)
                        if signals:
                            return (
                                f"Trading Signals for {signals['symbol']}:\n"
                                f"Current Price: ₹{signals['current_price']:.2f}\n"
                                f"RSI Signal: {signals['signals']['RSI_Signal']}\n"
                                f"MACD Signal: {signals['signals']['MACD_Signal']}\n"
                                f"Bollinger Bands Signal: {signals['signals']['BB_Signal']}\n"
                                f"Overall Signal: {signals['overall_signal']}"
                            )
                except Exception as e:
                    st.error(f"Error getting trading signals: {str(e)}")
            
            # Check for market terms query
            if any(word in query for word in ["what is", "explain", "define", "meaning of", "tell me about"]):
                try:
                    # Use ML chatbot's market terms
                    for term, explanation in self.ml_chatbot.market_terms.items():
                        if term in query:
                            return f"{term.upper()}: {explanation}"
                except:
                    pass
            
            # Check for watchlist query
            if "watchlist" in query:
                if self.watchlist:
                    return f"Your watchlist contains: {', '.join(self.watchlist)}"
                return "Your watchlist is empty"
            
            # Check for portfolio query
            if "portfolio" in query:
                if self.portfolio["holdings"]:
                    holdings = []
                    for symbol, data in self.portfolio["holdings"].items():
                        info = self.get_stock_info(symbol)
                        if info:
                            holdings.append(f"{info['name']}: {data['quantity']} shares")
                    return f"Your portfolio contains: {', '.join(holdings)}"
                return "Your portfolio is empty"
            
            # If no specific response is generated, try ML chatbot's detailed response
            try:
                intent, confidence, sentiment = self.ml_chatbot.classify_intent(query)
                return self.ml_chatbot.generate_detailed_response(intent, None, sentiment)
            except:
                return "I'm not sure about that. You can ask me about stock prices, market activity, sentiment analysis, your watchlist, or portfolio."
            
        except Exception as e:
            return f"I encountered an error: {str(e)}"

def show_chat():
    """Display the AI chat page with futuristic UI"""
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
            "Explain what is IPO?",
            "Show sector performance",
            "What are the trading signals for HDFC Bank?",
            "Show my portfolio analysis"
        ]
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Enhanced CSS with dashboard-like styling
    st.markdown("""
    <style>
        /* Global styles */
        .main {
            background: #FFFFFF;
            color: #1E293B;
        }
        
        /* Chat container */
        .chat-container {
            background: #FFFFFF;
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        /* Message bubbles */
        .chat-message {
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
            max-width: 85%;
            animation: messageSlide 0.3s ease-out;
        }
        
        .chat-message.user {
            background: #F1F5F9;
            margin-left: auto;
            border-bottom-right-radius: 0.25rem;
            color: #1E293B;
        }
        
        .chat-message.bot {
            background: #FFFFFF;
            margin-right: auto;
            border-bottom-left-radius: 0.25rem;
            border: 1px solid #E2E8F0;
            color: #1E293B;
        }
        
        .chat-message .content {
            font-size: 1rem;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        
        .chat-message .timestamp {
            font-size: 0.75rem;
            color: #64748B;
            margin-top: 0.5rem;
            text-align: right;
        }
        
        /* Input area */
        .stTextInput>div>div>input {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 0.5rem;
            padding: 0.75rem 1rem;
            color: #1E293B;
            font-size: 1rem;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #3B82F6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
            outline: none;
        }
        
        /* Button styling */
        .stButton>button {
            background: #3B82F6;
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .stButton>button:hover {
            background: #2563EB;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Market cards */
        .market-card {
            background: #FFFFFF;
            border-radius: 0.5rem;
            padding: 1.25rem;
            margin-bottom: 1rem;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .market-card:hover {
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .market-card .title {
            color: #64748B;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .market-card .value {
            color: #1E293B;
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 0.25rem;
        }
        
        .market-card .change {
            font-size: 1rem;
            font-weight: 500;
        }
        
        /* Typing indicator */
        .typing-indicator {
            display: flex;
            padding: 1rem;
            background: #F8FAFC;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            width: fit-content;
            border: 1px solid #E2E8F0;
        }
        
        .typing-indicator span {
            height: 6px;
            width: 6px;
            background: #3B82F6;
            border-radius: 50%;
            display: inline-block;
            margin: 0 2px;
            animation: typingPulse 1.4s infinite;
        }
        
        /* Help section */
        .help-section {
            background: #FFFFFF;
            border-radius: 0.5rem;
            padding: 1.5rem;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .help-section h3 {
            color: #1E293B;
            font-size: 1.1rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        .help-section ul {
            color: #475569;
            padding-left: 0;
            list-style-type: none;
        }
        
        .help-section li {
            margin-bottom: 0.75rem;
            padding-left: 1.25rem;
            position: relative;
        }
        
        .help-section li::before {
            content: '•';
            position: absolute;
            left: 0;
            color: #3B82F6;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            border-top: 1px solid #E2E8F0;
            color: #64748B;
            font-size: 0.875rem;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: #F1F5F9;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #CBD5E1;
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #94A3B8;
        }
        
        /* Section headers */
        h1, h2, h3 {
            color: #1E293B;
            font-weight: 600;
        }
        
        /* Links and interactive elements */
        a {
            color: #3B82F6;
            text-decoration: none;
        }
        
        a:hover {
            color: #2563EB;
        }
        
        /* Status colors */
        .positive {
            color: #059669;
        }
        
        .negative {
            color: #DC2626;
        }
        
        .neutral {
            color: #64748B;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main layout
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='color: #1E293B; font-size: 2rem; margin-bottom: 0.5rem;'>AI Stock Market Assistant</h1>
        <p style='color: #64748B; font-size: 1rem;'>
            Your intelligent companion for real-time market analysis and insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create main columns with adjusted ratio
    main_col1, main_col2 = st.columns([2, 1])

    with main_col1:
        # Chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Chat history display
        for user_msg, bot_msg in st.session_state.history:
            st.markdown(f"""
            <div class="chat-message user">
                <div class="content">{user_msg}</div>
                <div class="timestamp">{datetime.now().strftime("%H:%M")}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="chat-message bot">
                <div class="content">{bot_msg}</div>
                <div class="timestamp">{datetime.now().strftime("%H:%M")}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.processing:
            st.markdown("""
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # User input and buttons
        user_input = st.text_input("Ask your question:", key="user_input", 
                                 placeholder="Type your query about stocks, market activity, or analysis...")

        input_col1, input_col2 = st.columns([4, 1])
        with input_col1:
            if st.button("Send", key="send_button") or (user_input and st.session_state.get('last_input') != user_input):
                if user_input.strip():
                    st.session_state.processing = True
                    st.experimental_rerun()
        with input_col2:
            if st.button("Clear", key="clear_button"):
                st.session_state.history = []
                st.session_state.last_input = ""
                st.experimental_rerun()

        # Quick suggestions
        st.markdown("### Quick Suggestions")
        suggestion_cols = st.columns(4)
        for i, suggestion in enumerate(st.session_state.suggestions):
            if suggestion_cols[i % 4].button(suggestion, key=f"suggestion_{i}"):
                st.session_state.user_input = suggestion
                st.session_state.processing = True
                st.experimental_rerun()

    with main_col2:
        # Market Overview
        st.markdown("### Market Overview")
        try:
            market_data = st.session_state.chatbot.get_market_activity()
            if market_data:
                st.markdown('<div style="display: flex; flex-direction: column; gap: 0.75rem;">', unsafe_allow_html=True)
                
                # Market cards with updated styling
                for index, data in [
                    ("Nifty 50", market_data['nifty']),
                    ("Sensex", market_data['sensex']),
                    ("Market Status", {"value": market_data['market_status'], "change": market_data['last_updated']}),
                    ("Advance-Decline", market_data['advance_decline'])
                ]:
                    if index == "Market Status":
                        status_color = "#059669" if data['value'] == "Open" else "#DC2626"
                        st.markdown(f"""
                        <div class="market-card">
                            <div class="title">{index}</div>
                            <div class="value" style="color: {status_color}">{data['value']}</div>
                            <div class="change">Last Updated: {data['change']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif index == "Advance-Decline":
                        st.markdown(f"""
                        <div class="market-card">
                            <div class="title">{index} Ratio</div>
                            <div class="value">{data['ratio']:.2f}</div>
                            <div class="change">
                                Advances: {data['advances']} | Declines: {data['declines']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        change_color = "#059669" if data['change_pct'] >= 0 else "#DC2626"
                        st.markdown(f"""
                        <div class="market-card">
                            <div class="title">{index}</div>
                            <div class="value">₹{data['current']:,.2f}</div>
                            <div class="change" style="color: {change_color}">
                                {data['change_pct']:+.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error("Unable to fetch market data")

    # Bottom section with divider
    st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
    
    # Recent Searches and Help & Tips in a single row
    bottom_col1, bottom_col2 = st.columns([1, 2])

    with bottom_col1:
        st.markdown("### Recent Searches")
        if st.session_state.history:
            recent_searches = [msg[0] for msg in st.session_state.history[-5:]]
            for idx, search in enumerate(recent_searches):
                if st.button(search, key=f"recent_search_{idx}", use_container_width=True):
                    st.session_state.user_input = search
                    st.session_state.processing = True
                    st.experimental_rerun()
        else:
            st.info("No recent searches")

    with bottom_col2:
        st.markdown("### Help & Tips")
        st.markdown("""
        <div class="help-section">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
                <div>
                    <h3>Example Queries</h3>
                    <ul>
                        <li>Stock Price: "What's the price of TCS?"</li>
                        <li>Market Activity: "Show market activity"</li>
                        <li>Sentiment: "What's the sentiment for Reliance?"</li>
                        <li>Watchlist: "Show my watchlist"</li>
                    </ul>
                </div>
                <div>
                    <h3>Tips</h3>
                    <ul>
                        <li>Be specific with company names</li>
                        <li>Use stock symbols for faster results</li>
                        <li>Ask about market terms for explanations</li>
                        <li>Use the quick suggestions for common queries</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>
            Powered by Advanced AI • Real-time Market Data • Last Updated: {}
        </p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    show_chat() 