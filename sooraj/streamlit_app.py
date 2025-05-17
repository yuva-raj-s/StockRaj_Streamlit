import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Set page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Analysis Dashboard")
st.markdown("""
This dashboard provides technical analysis for Indian stocks using various indicators including:
- Moving Averages (SMA, EMA)
- MACD
- RSI
- Volume Analysis
- Fibonacci Levels
- Future Price Projections
""")

# Sidebar for input
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE, TCS)", "RELIANCE")
period = st.sidebar.selectbox(
    "Select Time Period",
    ["1y", "2y", "5y", "max"],
    index=1
)

# API endpoint
API_URL = "http://localhost:8000/analyze-stock"

def fetch_stock_data():
    try:
        response = requests.post(
            API_URL,
            json={"ticker": ticker, "period": period}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Main content
if st.sidebar.button("Analyze Stock"):
    with st.spinner("Analyzing stock data..."):
        data = fetch_stock_data()
        
        if data:
            # Display current analysis
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"₹{data['data']['analysis']['current_price']:.2f}")
            with col2:
                st.metric("Trend", data['data']['analysis']['trend'].upper())
            with col3:
                st.metric("Signal", data['data']['analysis']['current_signal'])

            # Create candlestick chart
            df = pd.DataFrame(data['data']['candlestick'])
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Add indicators
            sma_df = pd.DataFrame(data['data']['indicators']['sma_20'])
            ema_df = pd.DataFrame(data['data']['indicators']['ema_50'])
            future_df = pd.DataFrame(data['data']['indicators']['future_projection'])
            
            # Create figure
            fig = go.Figure()
            
            # Add candlestick
            fig.add_trace(go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ))
            
            # Add SMA
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sma_df['Date']),
                y=sma_df['SMA_20'],
                name='SMA 20',
                line=dict(color='blue', width=1)
            ))
            
            # Add EMA
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(ema_df['Date']),
                y=ema_df['EMA_50'],
                name='EMA 50',
                line=dict(color='purple', width=1)
            ))
            
            # Add future projection
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(future_df['Date']),
                y=future_df['Projection_Line'],
                name='Future Projection',
                line=dict(color='green', width=1, dash='dash')
            ))
            
            # Add buy/sell signals
            signals = pd.DataFrame(data['data']['signals'])
            if not signals.empty:
                signals['Date'] = pd.to_datetime(signals['Date'])
                buy_signals = signals[signals['Signal'] == 'BUY']
                sell_signals = signals[signals['Signal'] == 'SELL']
                
                fig.add_trace(go.Scatter(
                    x=buy_signals['Date'],
                    y=buy_signals['Close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ))
                
                fig.add_trace(go.Scatter(
                    x=sell_signals['Date'],
                    y=sell_signals['Close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))
            
            # Update layout
            fig.update_layout(
                title=f"{ticker} Stock Analysis",
                yaxis_title="Price (₹)",
                xaxis_title="Date",
                template="plotly_white",
                height=600
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display Fibonacci levels
            st.subheader("Fibonacci Levels")
            fib_levels = data['data']['fibonacci_levels']
            fib_df = pd.DataFrame(list(fib_levels.items()), columns=['Level', 'Price'])
            fib_df['Price'] = fib_df['Price'].round(2)
            st.dataframe(fib_df, use_container_width=True)
            
            # Display recent signals
            st.subheader("Recent Trading Signals")
            if not signals.empty:
                st.dataframe(signals.tail(5), use_container_width=True)
            else:
                st.info("No recent trading signals available")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and FastAPI") 