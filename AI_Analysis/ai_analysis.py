import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import sys

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import logging
import os
from .advanced_charting.advanced_chart import plot_advanced_chart
from .Current_Data.current_data import display_current_data
from .Prediction_Signals.prediction_signals import display_prediction_signals
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import plotly.graph_objects as go
from datetime import datetime, timedelta
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import json
from AI_Analysis.Current_Data.current_data import display_current_data
from SentimentAnalysis.sentiment_analysis import analyze_asset_sentiment as analyze_sentiment
from lstm_model.lstm_prediction import main as show_lstm_analysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_technical_indicators(df):
    """Calculate technical indicators for a stock"""
    try:
        # RSI
        rsi = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi.rsi()
        
        # MACD
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['Signal'] = macd.macd_signal()
        df['Histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        
        # Moving Averages
        sma = SMAIndicator(close=df['Close'], window=20)
        df['SMA_20'] = sma.sma_indicator()
        
        ema = EMAIndicator(close=df['Close'], window=20)
        df['EMA_20'] = ema.ema_indicator()
        
        # Volume indicators
        volume_sma = SMAIndicator(close=df['Volume'], window=20)
        df['Volume_SMA'] = volume_sma.sma_indicator()
        
        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return None

def get_stock_suggestions(query):
    try:
        # Add proper stock exchange suffixes
        exchanges = ['.NS', '.BO', '.NSE', '.BSE']  # For Indian stocks
        suggestions = []
        
        # Try with different exchange suffixes
        for suffix in exchanges:
            try:
                ticker = yf.Ticker(query + suffix)
                info = ticker.info
                if info and 'regularMarketPrice' in info:
                    suggestions.append({
                        'symbol': query + suffix,
                        'name': info.get('longName', info.get('shortName', query + suffix)),
                        'shortname': info.get('shortName', ''),
                        'sector': info.get('sector', 'N/A'),
                        'market_cap': info.get('marketCap', 0)
                    })
            except:
                continue
                
        # Try without suffix for international stocks
        try:
            ticker = yf.Ticker(query)
            info = ticker.info
            if info and 'regularMarketPrice' in info:
                suggestions.append({
                    'symbol': query,
                    'name': info.get('longName', info.get('shortName', query)),
                    'shortname': info.get('shortName', ''),
                    'sector': info.get('sector', 'N/A'),
                    'market_cap': info.get('marketCap', 0)
                })
        except:
            pass
            
        return suggestions
    except Exception as e:
        logger.error(f"Error getting suggestions: {str(e)}")
        return []

def get_stock_data(symbol, period="6mo", interval="1d"):
    """Get stock data with technical indicators"""
    try:
        # Format the symbol for Indian stocks
        if not symbol.endswith(('.NS', '.BO', '.NSE', '.BSE')):
            symbol = f"{symbol}.NS"
        
        # Get historical data with proper interval
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, prepost=True)  # Added prepost for real-time data
        
        if df.empty:
            # Try alternative exchanges if .NS fails
            for suffix in ['.BO', '.NSE', '.BSE']:
                if not symbol.endswith(suffix):
                    alt_symbol = f"{symbol.split('.')[0]}{suffix}"
                    alt_ticker = yf.Ticker(alt_symbol)
                    alt_df = alt_ticker.history(period=period, interval=interval, prepost=True)
                    if not alt_df.empty:
                        df = alt_df
                        break
            
            if df.empty:
                st.error(f"No data available for {symbol}. Please check the symbol and try again.")
                return None
        
        # Calculate technical indicators
        try:
            # RSI
            rsi = RSIIndicator(close=df['Close'], window=14)
            df['RSI'] = rsi.rsi()
            
            # MACD
            macd = MACD(close=df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = BollingerBands(close=df['Close'])
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower'] = bb.bollinger_lband()
            
            # Moving Averages
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            
            # Generate signals
            df['Signal'] = 0  # Initialize signal column
            
            # MACD Signal
            df.loc[(df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)), 'Signal'] = 1
            df.loc[(df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)), 'Signal'] = -1
            
            # RSI Signal
            df.loc[df['RSI'] < 30, 'Signal'] = 1  # Oversold
            df.loc[df['RSI'] > 70, 'Signal'] = -1  # Overbought
            
            # Bollinger Bands Signal
            df.loc[df['Close'] < df['BB_Lower'], 'Signal'] = 1  # Price below lower band
            df.loc[df['Close'] > df['BB_Upper'], 'Signal'] = -1  # Price above upper band
            
        except Exception as e:
            st.warning(f"Some technical indicators could not be calculated: {str(e)}")
        
        return df
    except Exception as e:
        st.error(f"Error getting stock data: {str(e)}")
        return None

def get_timeframe_options():
    return {
        "1D": ("1d", "5m"),
        "1W": ("5d", "15m"),
        "1M": ("1mo", "30m"),
        "3M": ("3mo", "1h"),
        "1Y": ("1y", "1d"),
        "All": ("max", "1d")
    }

def plot_technical_analysis(df, symbol):
    """Plot technical analysis charts"""
    try:
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_Upper'],
            name='BB Upper',
            line=dict(color='rgba(250, 0, 0, 0.3)')
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_Lower'],
            name='BB Lower',
            line=dict(color='rgba(0, 250, 0, 0.3)'),
            fill='tonexty'
        ))
        
        # Add Moving Averages
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_20'],
            name='SMA 20',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA_20'],
            name='EMA 20',
            line=dict(color='orange')
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Technical Analysis",
            yaxis_title="Price",
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Error plotting technical analysis: {str(e)}")
        return None

def get_sentiment_analysis(symbol):
    """Get sentiment analysis for a stock"""
    try:
        # Get news articles
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            return None
        
        # Analyze sentiment
        sentiments = []
        for article in news[:10]:  # Analyze last 10 articles
            if 'title' in article:
                blob = TextBlob(article['title'])
                sentiments.append(blob.sentiment.polarity)
        
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            sentiment_label = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
            
            return {
                'average_sentiment': avg_sentiment,
                'sentiment_label': sentiment_label,
                'article_count': len(sentiments)
            }
        return None
    except Exception as e:
        st.error(f"Error getting sentiment analysis: {str(e)}")
        return None

def get_trading_signals(df):
    """Generate trading signals based on technical indicators"""
    try:
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
        
        # Moving Average signals
        if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] and df['Close'].iloc[-2] <= df['SMA_20'].iloc[-2]:
            signals.append("Price crossed above 20-day SMA")
        elif df['Close'].iloc[-1] < df['SMA_20'].iloc[-1] and df['Close'].iloc[-2] >= df['SMA_20'].iloc[-2]:
            signals.append("Price crossed below 20-day SMA")
        
        return signals
    except Exception as e:
        st.error(f"Error generating trading signals: {str(e)}")
        return []

def get_price_metrics(data):
    current_price = data['Close'].iloc[-1]
    open_price = data['Open'].iloc[-1]
    high_price = data['High'].iloc[-1]
    low_price = data['Low'].iloc[-1]
    price_change = current_price - open_price
    price_change_pct = (price_change / open_price) * 100
    return current_price, open_price, high_price, low_price, price_change, price_change_pct

def show_technical_indicators(data, info=None):
    st.subheader("Technical Indicators")
    cols = st.columns(4)
    def safe_metric(val):
        if pd.isna(val) or val is None:
            return "N/A"
        try:
            return f"{val:.2f}"
        except Exception:
            return str(val)
    with cols[0]:
        st.metric("RSI", safe_metric(data['RSI'].iloc[-1]) if 'RSI' in data.columns else "N/A")
        st.metric("MACD", safe_metric(data['MACD'].iloc[-1]) if 'MACD' in data.columns else "N/A")
    with cols[1]:
        st.metric("SMA 20", safe_metric(data['SMA_20'].iloc[-1]) if 'SMA_20' in data.columns else "N/A")
        st.metric("SMA 50", safe_metric(data['SMA_50'].iloc[-1]) if 'SMA_50' in data.columns else "N/A")
    with cols[2]:
        st.metric("BB Upper", safe_metric(data['BB_Upper'].iloc[-1]) if 'BB_Upper' in data.columns else "N/A")
        st.metric("BB Lower", safe_metric(data['BB_Lower'].iloc[-1]) if 'BB_Lower' in data.columns else "N/A")
    with cols[3]:
        if info:
            st.metric("P/E Ratio", safe_metric(info.get('trailingPE', None)))
            st.metric("P/B Ratio", safe_metric(info.get('priceToBook', None)))
        else:
            st.metric("P/E Ratio", "N/A")
            st.metric("P/B Ratio", "N/A")

def show_technical_signals(data):
    st.subheader("Trading Signal")
    if 'Signal' in data.columns:
        last_signal = data['Signal'].iloc[-1]
        signal_text = "BUY" if last_signal == 1 else "SELL" if last_signal == -1 else "NEUTRAL"
        st.info(f"Current Signal: {signal_text}")
    else:
        st.warning("Technical signals not available.")

def show_sentiment_section(symbol):
    st.header("Market Sentiment & News Analysis")
    try:
        articles_df, sentiment_summary, sentiment_bar_img, sentiment_gauge, stock_chart, ticker = analyze_asset_sentiment(symbol)
        if articles_df is None or articles_df.empty:
            st.warning("No news articles found for this symbol.")
            return
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Market Sentiment")
            st.markdown(sentiment_summary)
            st.image(sentiment_bar_img)
        with col2:
            st.markdown("### Sentiment Gauge")
            st.image(sentiment_gauge)
        st.markdown("### Articles and Sentiment Analysis")
        st.dataframe(articles_df, use_container_width=True)
    except Exception as e:
        st.error(f"Sentiment analysis error: {e}")

def show_ai_analysis():
    st.title("🤖 AI Analysis")
    tab1, tab2, tab3 = st.tabs(["Current Market", "Prediction & Signals", "Sentiment Analysis"])

    # --- Tab 1: Current Market ---
    with tab1:
        # Search and Controls Section - 2 columns like tickertape/moneycontrol
        col1, col2 = st.columns([3, 1])
        with col1:
            symbol = st.text_input("Enter stock symbol", "RELIANCE", key="ai_analysis_symbol1",
                                 help="Enter NSE stock symbol (e.g., RELIANCE, TCS, INFY)")
        with col2:
            timeframe = st.selectbox("Timeframe", ["1D", "1W", "1M", "3M", "1Y", "ALL"], 
                                   key="ai_analysis_timeframe1")

        # Timeframe mapping with proper intervals
        period_map = {
            "1D": ("1d", "5m"),    # Changed to 5m for more granular intraday data
            "1W": ("5d", "15m"),
            "1M": ("1mo", "30m"),
            "3M": ("3mo", "1d"),
            "1Y": ("1y", "1d"),
            "ALL": ("max", "1d")
        }
        period, interval = period_map[timeframe]

        if symbol:
            try:
                data = get_stock_data(symbol, period=period, interval=interval)
                if data is not None and not data.empty:
                    # Stock Chart & Price Analysis Section
                    st.markdown("""
                    <div style='background-color: #1e2130; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                        <h3 style='color: #ffffff; margin: 0 0 1rem 0;'>Stock Chart & Price Analysis</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    # Price metrics with improved styling
                    current_price, open_price, high_price, low_price, price_change, price_change_pct = get_price_metrics(data)
                    cols = st.columns(6)
                    metrics = [
                        ('Open', f'₹{open_price:.2f}'),
                        ('High', f'₹{high_price:.2f}'),
                        ('Low', f'₹{low_price:.2f}'),
                        ('Current', f'₹{current_price:.2f}'),
                        ('Change', f'₹{price_change:.2f}'),
                        ('Change %', f'{price_change_pct:+.2f}%')
                    ]
                    
                    for col, (label, value) in zip(cols, metrics):
                        with col:
                            st.metric(label, value, 
                                    delta=f"{price_change_pct:+.2f}%" if label == "Change %" else None,
                                    delta_color="normal")

                    # Display chart
                    display_current_data(data, "Candles")

                    # # Technical Analysis Section
                    # st.markdown("""
                    # <div style='background-color: #1e2130; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                    #     <h3 style='color: #ffffff; margin: 0 0 1rem 0;'>Technical Analysis</h3>
                    # </div>
                    # """, unsafe_allow_html=True)

                    # # Display technical indicators in a grid
                    # col1, col2, col3, col4 = st.columns(4)
                    # indicators = [
                    #     ('RSI', data['RSI'].iloc[-1] if 'RSI' in data.columns else None),
                    #     ('MACD', data['MACD'].iloc[-1] if 'MACD' in data.columns else None),
                    #     ('SMA 20', data['SMA_20'].iloc[-1] if 'SMA_20' in data.columns else None),
                    #     ('EMA 20', data['EMA_20'].iloc[-1] if 'EMA_20' in data.columns else None)
                    # ]
                    
                    # for col, (label, value) in zip([col1, col2, col3, col4], indicators):
                    #     with col:
                    #         if value is not None and not pd.isna(value):
                    #             st.metric(label, f"{value:.2f}")
                    #         else:
                    #             st.metric(label, "N/A")

                    # Trading Signal with color-coded display
                    if 'Signal' in data.columns:
                        current_signal = data['Signal'].iloc[-1]
                        signal_text = "BUY" if current_signal == 1 else "SELL" if current_signal == -1 else "NEUTRAL"
                        signal_color = "green" if current_signal == 1 else "red" if current_signal == -1 else "gray"
                        
                        st.markdown(f"""
                        <div style='background-color: #1e2130; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                            <h3 style='color: #ffffff; margin: 0 0 0.5rem 0;'>Trading Signal</h3>
                            <p style='color: {signal_color}; font-size: 1.5rem; font-weight: bold; margin: 0;'>{signal_text}</p>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error fetching/displaying data: {e}")

    # --- Tab 2: Prediction & Signals ---
    with tab2:
        st.header("Prediction & Signals (LSTM)")
        show_lstm_analysis()

    # --- Tab 3: Sentiment Analysis ---
    with tab3:
        st.header("Sentiment Analysis & News")
        symbol3 = st.text_input("Enter stock symbol (e.g., RELIANCE, TCS):", "RELIANCE", key="ai_analysis_symbol3")
        if symbol3 and st.button("Analyze Sentiment", key="analyze_sentiment_btn"):
            show_sentiment_section(symbol3)

if __name__ == "__main__":
    show_ai_analysis() 