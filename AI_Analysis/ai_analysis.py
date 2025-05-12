import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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
                        'name': info.get('shortName', query + suffix),
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
                    'name': info.get('shortName', query),
                    'sector': info.get('sector', 'N/A'),
                    'market_cap': info.get('marketCap', 0)
                })
        except:
            pass
            
        return suggestions
    except Exception as e:
        logger.error(f"Error getting suggestions: {str(e)}")
        return []

def get_stock_data(symbol, period="6mo"):
    """Get stock data with technical indicators"""
    try:
        # Add .NS suffix if not present
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        
        # Get historical data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            st.error(f"No data available for {symbol}")
            return None
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
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

def show_ai_analysis():
    """Display the AI analysis page"""
    st.title("🤖 AI Analysis")
    
    # Stock selection
    symbol = st.text_input("Enter stock symbol (e.g., RELIANCE, TCS):", "RELIANCE")
    
    if symbol:
        # Get stock data
        df = get_stock_data(symbol)
        
        if df is not None:
            # Technical Analysis
            st.header("📊 Technical Analysis")
            
            # Plot technical analysis
            fig = plot_technical_analysis(df, symbol)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Trading Signals
            st.header("🎯 Trading Signals")
            signals = get_trading_signals(df)
            if signals:
                for signal in signals:
                    st.info(signal)
            else:
                st.info("No significant trading signals detected")
            
            # Technical Indicators
            st.header("📈 Technical Indicators")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}")
                st.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
                st.metric("Signal Line", f"{df['Signal'].iloc[-1]:.2f}")
            
            with col2:
                st.metric("20-day SMA", f"{df['SMA_20'].iloc[-1]:.2f}")
                st.metric("20-day EMA", f"{df['EMA_20'].iloc[-1]:.2f}")
                st.metric("Volume SMA", f"{df['Volume_SMA'].iloc[-1]:.0f}")
            
            # Sentiment Analysis
            st.header("😊 Sentiment Analysis")
            sentiment = get_sentiment_analysis(symbol)
            if sentiment:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Sentiment",
                        sentiment['sentiment_label'],
                        f"{sentiment['average_sentiment']:.2f}"
                    )
                
                with col2:
                    st.metric("Articles Analyzed", sentiment['article_count'])
                
                with col3:
                    # Sentiment gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=sentiment['average_sentiment'],
                        title={'text': "Sentiment Score"},
                        gauge={
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [-1, -0.5], 'color': "red"},
                                {'range': [-0.5, 0], 'color': "orange"},
                                {'range': [0, 0.5], 'color': "lightgreen"},
                                {'range': [0.5, 1], 'color': "green"}
                            ]
                        }
                    ))
                    fig.update_layout(height=200)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No sentiment data available")
        else:
            st.error("Unable to fetch stock data")

if __name__ == "__main__":
    show_ai_analysis() 