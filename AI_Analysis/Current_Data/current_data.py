import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

def get_timeframe_options():
    return {
        "1D": ("1d", "5m"),
        "1W": ("5d", "15m"),
        "1M": ("1mo", "30m"),
        "3M": ("3mo", "1h"),
        "1Y": ("1y", "1d"),
        "All": ("max", "1d")
    }

def plot_simple_chart(data, chart_type="line"):
    """Create a simple price chart based on the selected chart type"""
    fig = go.Figure()
    
    if chart_type.lower() == "line":
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                name='Price', line=dict(color='blue')))
    elif chart_type.lower() == "area":
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                name='Price', fill='tozeroy', line=dict(color='blue')))
    elif chart_type.lower() == "baseline":
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                name='Price', line=dict(color='blue')))
        fig.add_hline(y=data['Close'].iloc[0], line_dash="dash", line_color="red")
    elif chart_type.lower() == "candles":
        fig.add_trace(go.Candlestick(x=data.index,
                                    open=data['Open'],
                                    high=data['High'],
                                    low=data['Low'],
                                    close=data['Close'],
                                    name='Price'))
    
    fig.update_layout(
        title='Price Chart',
        yaxis_title='Price (₹)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    return fig

def format_price(price):
    return f"₹{price:,.2f}"

def display_current_data(data, chart_type):
    try:
        # Display chart type selector
        chart_type = st.selectbox("Select Chart Type", ["Line", "Area", "Baseline", "Candles"], index=0, key="current_data_chart_type")
        
        # Display the chart
        st.plotly_chart(plot_simple_chart(data, chart_type), use_container_width=True)
        
        # Display current price information
        current_price = data['Close'].iloc[-1]
        open_price = data['Open'].iloc[-1]
        high_price = data['High'].iloc[-1]
        low_price = data['Low'].iloc[-1]
        price_change = current_price - open_price
        price_change_pct = (price_change / open_price) * 100
        
        # Price information in columns
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Open", format_price(open_price))
        with col2:
            st.metric("High", format_price(high_price))
        with col3:
            st.metric("Low", format_price(low_price))
        with col4:
            st.metric("Current", format_price(current_price))
        with col5:
            st.metric("Change", format_price(price_change))
        with col6:
            st.metric("Change %", f"{price_change_pct:.2f}%")
        
        # Display technical indicators
        st.subheader("Technical Indicators")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RSI", f"{data['RSI'].iloc[-1]:.2f}")
            st.metric("MACD", f"{data['MACD'].iloc[-1]:.2f}")
        with col2:
            st.metric("SMA 20", format_price(data['SMA_20'].iloc[-1]))
            st.metric("SMA 50", format_price(data['SMA_50'].iloc[-1]))
        with col3:
            st.metric("EMA 20", format_price(data['EMA_20'].iloc[-1]))
            st.metric("BB Upper", format_price(data['BB_Upper'].iloc[-1]))
        with col4:
            st.metric("BB Lower", format_price(data['BB_Lower'].iloc[-1]))
            st.metric("BB Middle", format_price(data['BB_Middle'].iloc[-1]))
            
    except Exception as e:
        logger.error(f"Error displaying current data: {str(e)}")
        st.error(f"Error displaying current data: {str(e)}") 