import streamlit as st
import sys
import os
import time
from datetime import datetime
import pytz
import yfinance as yf
import pandas as pd

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_marquee_data(symbols):
    """Get live market data for marquee stocks"""
    try:
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
        st.error(f"Error fetching marquee data: {str(e)}")
        return None

def get_market_overview():
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
        st.error(f"Error fetching market overview: {str(e)}")
        return None

def get_top_indian_stocks(limit=10):
    """Get top performing Indian stocks"""
    try:
        # List of major Indian stocks
        symbols = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS'
        ]
        
        data = []
        for symbol in symbols[:limit]:
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
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error fetching top stocks: {str(e)}")
        return None

def fetch_financial_news():
    """Fetch latest financial news"""
    try:
        # This is a placeholder. In a real application, you would integrate with a news API
        return [
            {
                'headline': 'Market Update: Indian stocks trade higher',
                'source': 'Financial Express',
                'link': 'https://www.financialexpress.com',
                'published_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'headline': 'RBI keeps repo rate unchanged',
                'source': 'Economic Times',
                'link': 'https://economictimes.indiatimes.com',
                'published_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return None

def show_dashboard():
    # Title and last updated time
    st.title("📊 Market Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Marquee Section
    st.header("📈 Live Market Ticker")
    marquee_symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
    try:
        marquee_data = get_marquee_data(marquee_symbols)
        if marquee_data:
            # Create a horizontal scrolling container for marquee
            marquee_container = st.container()
            with marquee_container:
                # Use a fixed number of columns
                cols = st.columns(5)  # Fixed 5 columns for 5 stocks
                for i, data in enumerate(marquee_data):
                    if i < 5:  # Ensure we don't exceed column count
                        with cols[i]:
                            st.metric(
                                label=f"{data['name']} ({data['symbol']})",
                                value=f"₹{data['current_price']}",
                                delta=f"{data['change_percent']}%",
                                delta_color="normal"
                            )
        else:
            st.warning("No data available for live market ticker.")
    except Exception as e:
        st.error(f"Error fetching live market data: {str(e)}")

    # Market Overview Section
    st.header("🌍 Market Overview")
    try:
        market_data = get_market_overview()
        if market_data:
            cols = st.columns(3)
            for i, (index, data) in enumerate(market_data.items()):
                with cols[i]:
                    st.metric(
                        label=index,
                        value=f"₹{data['price']}",
                        delta=f"{data['change_percent']}%",
                        delta_color="normal"
                    )
        else:
            st.warning("No data available for market overview.")
    except Exception as e:
        st.error(f"Error fetching market overview: {str(e)}")

    # Top Stocks Section
    st.header("🏆 Top Indian Stocks")
    try:
        top_stocks = get_top_indian_stocks(limit=10)
        if top_stocks is not None and not top_stocks.empty:
            st.dataframe(
                top_stocks,
                column_config={
                    "symbol": "Symbol",
                    "name": "Company",
                    "price": st.column_config.NumberColumn("Price (₹)", format="₹%.2f"),
                    "change_percent": st.column_config.NumberColumn("Change %", format="%.2f%%"),
                    "volume": st.column_config.NumberColumn("Volume", format="%d")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("No data available for top stocks.")
    except Exception as e:
        st.error(f"Error fetching top stocks: {str(e)}")

    # Financial News Section
    st.header("📰 Latest Financial News")
    try:
        news_data = fetch_financial_news()
        if news_data:
            for news in news_data:
                with st.expander(f"{news['headline']} - {news['source']}"):
                    st.markdown(f"[Read more]({news['link']})")
                    st.caption(f"Published: {news['published_date']}")
        else:
            st.warning("No news available at the moment.")
    except Exception as e:
        st.error(f"Error fetching financial news: {str(e)}")

if __name__ == "__main__":
    show_dashboard()

# Add auto-refresh
# time.sleep(15)  # Refresh every 15 seconds
# st.experimental_rerun() 