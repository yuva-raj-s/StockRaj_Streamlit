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
        """Process user query and generate response"""
        try:
            # Check for sentiment query
            if "sentiment" in query.lower():
                try:
                    # Extract stock symbol using ML chatbot's symbol detection
                    symbol = self.ml_chatbot.get_stock_symbol(query)
                    
                    if not symbol:
                        # Try to extract symbol from the query using common patterns
                        if "for" in query.lower():
                            parts = query.lower().split("for")
                            if len(parts) > 1:
                                potential_symbol = parts[1].strip().upper()
                                if potential_symbol in ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]:
                                    symbol = potential_symbol
                    
                    if symbol:
                        print(f"Processing sentiment analysis for {symbol}")
                        sentiment_data = self.ml_chatbot.get_sentiment_analysis(symbol)
                        
                        if sentiment_data:
                            # Format the response
                            response = f"📊 Sentiment Analysis for {symbol}:\n\n"
                            response += f"Current Price: ₹{sentiment_data['current_price']:,.2f} ({sentiment_data['price_change']:+.2f}%)\n"
                            response += f"Overall Sentiment: {sentiment_data['market_context']}\n"
                            response += f"Sentiment Score: {sentiment_data['sentiment_score']:.2f}\n\n"
                            
                            # News sentiment breakdown
                            response += "📰 News Sentiment Breakdown:\n"
                            response += f"Positive: {sentiment_data['positive']} articles\n"
                            response += f"Negative: {sentiment_data['negative']} articles\n"
                            response += f"Neutral: {sentiment_data['neutral']} articles\n"
                            response += f"Total News Analyzed: {sentiment_data['total_news']}\n\n"
                            
                            # Technical indicators
                            if sentiment_data['technical_indicators']['rsi'] is not None:
                                response += "📈 Technical Indicators:\n"
                                response += f"RSI: {sentiment_data['technical_indicators']['rsi']:.2f}\n"
                                response += f"MACD: {sentiment_data['technical_indicators']['macd']:.2f}\n"
                                response += f"MACD Signal: {sentiment_data['technical_indicators']['macd_signal']:.2f}\n\n"
                            
                            # Recent news
                            if sentiment_data['recent_news']:
                                response += "📰 Recent News:\n"
                                for news in sentiment_data['recent_news']:
                                    sentiment_emoji = "🟢" if news['sentiment'] > 0.1 else "🔴" if news['sentiment'] < -0.1 else "⚪"
                                    response += f"{sentiment_emoji} {news['title']}\n"
                                    response += f"   Source: {news['publisher']}\n"
                                    response += f"   Published: {news['published']}\n\n"
                            
                            return response
                        else:
                            print(f"Failed to get sentiment data for {symbol}")
                            return f"Unable to fetch sentiment analysis for {symbol} at the moment. Please try again later."
                    else:
                        return "Please specify which stock's sentiment you'd like to know about."
                except Exception as e:
                    print(f"Error in sentiment analysis: {str(e)}")
                    return f"Error analyzing sentiment: {str(e)}"
            
            # Check for sector performance query
            if any(word in query.lower() for word in ['sector', 'sectors', 'performance']):
                try:
                    from Market_Activity.Sector.sector_analysis import SectorAnalysis
                    sector_analysis = SectorAnalysis()
                    performance = sector_analysis.get_sector_performance()
                    
                    if performance:
                        response = "📊 Sector Performance:\n\n"
                        for sector, data in performance.items():
                            change_emoji = "🟢" if data['change_percent'] > 0 else "🔴" if data['change_percent'] < 0 else "⚪"
                            response += f"{change_emoji} {sector}:\n"
                            response += f"   Price: ₹{data['current_price']:,.2f}\n"
                            response += f"   Change: {data['change_percent']:+.2f}%\n"
                            response += f"   Volume: {data['volume']:,}\n\n"
                        return response
                    else:
                        return "Unable to fetch sector performance at the moment. Please try again later."
                except Exception as e:
                    print(f"Error in sector performance: {str(e)}")
                    return f"Error fetching sector performance: {str(e)}"
            
            # Process other queries
            return self.ml_chatbot.process_query(query)
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return "I'm having trouble understanding. Could you please rephrase your question?"

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

        # Quick suggestions (restored)
        st.markdown('<div style="margin-bottom: 1.2rem;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size:1.08rem; margin-bottom:0.5rem;">💡 Quick Suggestions</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, suggestion in enumerate(st.session_state.suggestions):
            if cols[i % 4].button(suggestion, key=f"suggestion_{i}", help="Click to use this suggestion"):
                st.session_state.user_input = suggestion
                st.experimental_rerun()
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
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        st.session_state.history.append((user_input, "I'm having trouble processing your request. Please try again."))
                        st.session_state['bot_loading'] = False
        st.markdown('</div>', unsafe_allow_html=True)
        # Clear chat button
        if st.button("Clear Chat", key="clear_button"):
            st.session_state.history = []
            st.session_state.last_input = ""
            st.experimental_rerun()

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
                    st.experimental_rerun()
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