import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
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

def format_price(price):
    return f"₹{price:,.2f}"

def plot_advanced_chart(data, chart_type="line"):
    """Create an advanced chart with multiple indicators"""
    # Create figure with secondary y-axis
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       row_heights=[0.5, 0.25, 0.25])

    # Add price chart based on type
    if chart_type.lower() == "line":
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                name='Price', 
                                line=dict(color='blue', 
                                         width=2,
                                         shape='spline',  # Makes the line smoother
                                         smoothing=0.3)),  # Controls the smoothness
                     row=1, col=1)
    elif chart_type.lower() == "area":
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                name='Price', 
                                fill='tozeroy', 
                                line=dict(color='blue',
                                         width=2,
                                         shape='spline',
                                         smoothing=0.3)), 
                     row=1, col=1)
    elif chart_type.lower() == "baseline":
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                name='Price', 
                                line=dict(color='blue',
                                         width=2,
                                         shape='spline',
                                         smoothing=0.3)), 
                     row=1, col=1)
        fig.add_hline(y=data['Close'].iloc[0], 
                     line_dash="dash", 
                     line_color="red",
                     line_width=1,
                     row=1, col=1)
    elif chart_type.lower() == "candles":
        fig.add_trace(go.Candlestick(x=data.index,
                                    open=data['Open'],
                                    high=data['High'],
                                    low=data['Low'],
                                    close=data['Close'],
                                    name='Price',
                                    increasing_line_color='green',
                                    decreasing_line_color='red'), 
                     row=1, col=1)

    # Add moving averages if available
    if 'SMA_20' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], 
                                name='SMA 20', 
                                line=dict(color='orange',
                                         width=1.5,
                                         shape='spline',
                                         smoothing=0.3)), 
                     row=1, col=1)
    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], 
                                name='SMA 50', 
                                line=dict(color='purple',
                                         width=1.5,
                                         shape='spline',
                                         smoothing=0.3)), 
                     row=1, col=1)
    if 'EMA_20' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], 
                                name='EMA 20', 
                                line=dict(color='green',
                                         width=1.5,
                                         shape='spline',
                                         smoothing=0.3)), 
                     row=1, col=1)

    # Add Bollinger Bands if available
    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], 
                                name='BB Upper', 
                                line=dict(color='gray', 
                                         dash='dash',
                                         width=1,
                                         shape='spline',
                                         smoothing=0.3)), 
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], 
                                name='BB Lower', 
                                line=dict(color='gray', 
                                         dash='dash',
                                         width=1,
                                         shape='spline',
                                         smoothing=0.3)), 
                     row=1, col=1)

    # Add MACD if available
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns and 'MACD_Hist' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], 
                                name='MACD', 
                                line=dict(color='blue',
                                         width=1.5,
                                         shape='spline',
                                         smoothing=0.3)), 
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], 
                                name='Signal', 
                                line=dict(color='red',
                                         width=1.5,
                                         shape='spline',
                                         smoothing=0.3)), 
                     row=2, col=1)
        fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], 
                            name='Histogram',
                            marker_color='gray'), 
                     row=2, col=1)

    # Add RSI if available
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], 
                                name='RSI', 
                                line=dict(color='purple',
                                         width=1.5,
                                         shape='spline',
                                         smoothing=0.3)), 
                     row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=[70]*len(data), 
                                name='Overbought', 
                                line=dict(color='red', 
                                         dash='dash',
                                         width=1,
                                         shape='spline',
                                         smoothing=0.3)), 
                     row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=[30]*len(data), 
                                name='Oversold', 
                                line=dict(color='green', 
                                         dash='dash',
                                         width=1,
                                         shape='spline',
                                         smoothing=0.3)), 
                     row=3, col=1)

    # Add volume
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], 
                        name='Volume',
                        marker_color='gray',
                        opacity=0.5), 
                 row=1, col=1)

    # Update layout
    fig.update_layout(
        title='Advanced Chart Analysis',
        yaxis_title='Price (₹)',
        yaxis2_title='MACD',
        yaxis3_title='RSI',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Display price information
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

    # Display technical indicators if available
    st.subheader("Technical Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if 'RSI' in data.columns:
            st.metric("RSI", f"{data['RSI'].iloc[-1]:.2f}")
        if 'MACD' in data.columns:
            st.metric("MACD", f"{data['MACD'].iloc[-1]:.2f}")
    with col2:
        if 'SMA_20' in data.columns:
            st.metric("SMA 20", format_price(data['SMA_20'].iloc[-1]))
        if 'SMA_50' in data.columns:
            st.metric("SMA 50", format_price(data['SMA_50'].iloc[-1]))
    with col3:
        if 'EMA_20' in data.columns:
            st.metric("EMA 20", format_price(data['EMA_20'].iloc[-1]))
        if 'BB_Upper' in data.columns:
            st.metric("BB Upper", format_price(data['BB_Upper'].iloc[-1]))
    with col4:
        if 'BB_Lower' in data.columns:
            st.metric("BB Lower", format_price(data['BB_Lower'].iloc[-1]))
        if 'BB_Middle' in data.columns:
            st.metric("BB Middle", format_price(data['BB_Middle'].iloc[-1]))
            
    # Add Bollinger Bands information
    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns and 'BB_Middle' in data.columns:
        current_price = data['Close'].iloc[-1]
        bb_upper = data['BB_Upper'].iloc[-1]
        bb_lower = data['BB_Lower'].iloc[-1]
        bb_middle = data['BB_Middle'].iloc[-1]
        
        # Calculate Bollinger Band width
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        # Determine Bollinger Band position
        if current_price > bb_upper:
            bb_position = "Above Upper Band"
        elif current_price < bb_lower:
            bb_position = "Below Lower Band"
        else:
            bb_position = "Within Bands"
            
        st.subheader("Bollinger Bands Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Band Width", f"{bb_width:.4f}")
        with col2:
            st.metric("Position", bb_position)
        with col3:
            st.metric("Current Price vs Middle", 
                     f"{((current_price - bb_middle) / bb_middle * 100):.2f}%")

    return fig 