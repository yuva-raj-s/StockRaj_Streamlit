import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import json
import os
from textblob import TextBlob

class IndianStockChatbot:
    def __init__(self):
        """Initialize the chatbot"""
        self.market_status = self.get_market_status()
        self.watchlist = self.load_watchlist()
        self.portfolio = self.load_portfolio()
    
    def get_market_status(self):
        """Get current market status"""
        try:
            nifty = yf.Ticker("^NSEI")
            info = nifty.info
            
            if 'regularMarketTime' in info:
                market_time = datetime.fromtimestamp(info['regularMarketTime'])
                now = datetime.now()
                
                # Market hours: 9:15 AM to 3:30 PM IST
                market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
                
                if market_open <= now <= market_close:
                    return "Open"
                else:
                    return "Closed"
            return "Unknown"
        except Exception as e:
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
        """Get market activity data"""
        try:
            # Get Nifty 50 data
            nifty = yf.Ticker("^NSEI")
            nifty_info = nifty.info
            
            # Get Sensex data
            sensex = yf.Ticker("^BSESN")
            sensex_info = sensex.info
            
            if nifty_info and sensex_info:
                return {
                    'nifty': {
                        'current': nifty_info.get('regularMarketPrice', 0),
                        'change_pct': nifty_info.get('regularMarketChangePercent', 0)
                    },
                    'sensex': {
                        'current': sensex_info.get('regularMarketPrice', 0),
                        'change_pct': sensex_info.get('regularMarketChangePercent', 0)
                    },
                    'market_status': self.market_status,
                    'advance_decline': self.get_advance_decline_ratio()
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
            # Convert query to lowercase for easier matching
            query = query.lower()
            
            # Check for stock price query
            if "price" in query or "value" in query:
                # Extract stock symbol
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
                    return f"Market Status: {activity['market_status']}\nNifty 50: ₹{activity['nifty']['current']:,.2f} ({activity['nifty']['change_pct']:+.2f}%)\nSensex: ₹{activity['sensex']['current']:,.2f} ({activity['sensex']['change_pct']:+.2f}%)"
            
            # Check for sentiment query
            if "sentiment" in query:
                words = query.split()
                for word in words:
                    if word.upper() in ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]:
                        sentiment = self.get_sentiment(f"{word}.NS")
                        if sentiment:
                            return f"The sentiment for {word} is {sentiment['label']} (Score: {sentiment['average']:.2f})"
            
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
            
            # Default response for unrecognized queries
            return "I'm not sure about that. You can ask me about stock prices, market activity, sentiment analysis, your watchlist, or portfolio."
            
        except Exception as e:
            return f"I encountered an error: {str(e)}"

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
            "Explain what is IPO?",
            "Show sector performance",
            "What are the trading signals for HDFC Bank?",
            "Show my portfolio analysis"
        ]

    # Custom CSS
    st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stTextInput>div>div>input {
            font-size: 1.1rem;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #2b313e;
        }
        .chat-message.bot {
            background-color: #475063;
        }
        .chat-message .content {
            display: flex;
            margin-top: 0.5rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 0.5rem;
            height: 3rem;
            font-size: 1.1rem;
        }
        .suggestion-chip {
            display: inline-block;
            padding: 0.5rem 1rem;
            margin: 0.25rem;
            border-radius: 1rem;
            background-color: #2b313e;
            cursor: pointer;
        }
        .suggestion-chip:hover {
            background-color: #475063;
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
        st.title("💬 AI Chat Assistant")
        st.write("Ask about Indian stocks, indices, market activity, portfolio, sentiment, or watchlist.")

        # Chat history display with custom styling
        for user_msg, bot_msg in st.session_state.history:
            st.markdown(f"""
            <div class="chat-message user">
                <strong>You:</strong>
                <div class="content">{user_msg}</div>
            </div>
            <div class="chat-message bot">
                <strong>Bot:</strong>
                <div class="content">{bot_msg}</div>
            </div>
            """, unsafe_allow_html=True)

        # Quick suggestions
        st.markdown("### 💡 Quick Suggestions")
        cols = st.columns(4)
        for i, suggestion in enumerate(st.session_state.suggestions):
            if cols[i % 4].button(suggestion, key=f"suggestion_{i}"):
                st.session_state.user_input = suggestion
                st.experimental_rerun()

        # User input
        user_input = st.text_input("Type your question:", key="user_input")

        # Send button
        if st.button("Send", key="send_button") or (user_input and st.session_state.get('last_input') != user_input):
            if user_input.strip():
                with st.spinner("Processing your query..."):
                    try:
                        bot_response = st.session_state.chatbot.process_query(user_input)
                        st.session_state.history.append((user_input, bot_response))
                        st.session_state.last_input = user_input
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        st.session_state.history.append((user_input, "I'm having trouble processing your request. Please try again."))

        # Clear chat button
        if st.button("Clear Chat", key="clear_button"):
            st.session_state.history = []
            st.session_state.last_input = ""
            st.experimental_rerun()

    with col2:
        # Market Overview
        st.markdown("### 📊 Market Overview")
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
                status_color = "green" if market_data['market_status'] == "Open" else "red"
                st.markdown(f"Market Status: <span style='color:{status_color}'>{market_data['market_status']}</span>", unsafe_allow_html=True)
                
                # Advance-Decline Ratio
                if market_data['advance_decline']['ratio'] != float('inf'):
                    st.metric(
                        "Advance-Decline Ratio",
                        f"{market_data['advance_decline']['ratio']:.2f}",
                        f"Advances: {market_data['advance_decline']['advances']} | Declines: {market_data['advance_decline']['declines']}"
                    )
        except Exception as e:
            st.error("Unable to fetch market data")

        # Recent Searches
        st.markdown("### 🔍 Recent Searches")
        if st.session_state.history:
            recent_searches = [msg[0] for msg in st.session_state.history[-5:]]
            for search in recent_searches:
                if st.button(search, key=f"recent_{search}"):
                    st.session_state.user_input = search
                    st.experimental_rerun()
        else:
            st.info("No recent searches")

        # Help Section
        with st.expander("💡 Help & Tips"):
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

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Powered by AI • Real-time Market Data • Last Updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    show_chat() 