import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

def get_market_status():
    """Get current market status"""
    try:
        # Get NSE index data
        nifty = yf.Ticker("^NSEI")
        info = nifty.info
        
        # Check if market is open based on regularMarketTime
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
        st.error(f"Error getting market status: {str(e)}")
        return "Unknown"

def get_sector_performance():
    """Get sector-wise performance"""
    try:
        # List of sector indices
        sectors = {
            'NIFTY AUTO': 'NIFTYAUTO.NS',
            'NIFTY BANK': 'NIFTYBANK.NS',
            'NIFTY FMCG': 'NIFTYFMCG.NS',
            'NIFTY IT': 'NIFTYIT.NS',
            'NIFTY PHARMA': 'NIFTYPHARMA.NS',
            'NIFTY REALTY': 'NIFTYREALTY.NS'
        }
        
        data = []
        for name, symbol in sectors.items():
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info and 'regularMarketPrice' in info:
                data.append({
                    'Sector': name,
                    'Price': info['regularMarketPrice'],
                    'Change %': info.get('regularMarketChangePercent', 0)
                })
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error getting sector performance: {str(e)}")
        return None

def get_market_breadth():
    """Get market breadth data"""
    try:
        # Get NSE data
        nifty = yf.Ticker("^NSEI")
        info = nifty.info
        
        if 'regularMarketPrice' in info:
            # Get top gainers and losers
            gainers = yf.download("^NSEI", period="1d", interval="1m")
            if not gainers.empty:
                advances = len(gainers[gainers['Close'] > gainers['Open']])
                declines = len(gainers[gainers['Close'] < gainers['Open']])
                unchanged = len(gainers) - advances - declines
                
                return {
                    'advances': advances,
                    'declines': declines,
                    'unchanged': unchanged,
                    'ratio': advances / declines if declines > 0 else float('inf')
                }
        return None
    except Exception as e:
        st.error(f"Error getting market breadth: {str(e)}")
        return None

def get_top_gainers_losers():
    """Get top gainers and losers"""
    try:
        # List of major stocks
        symbols = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS'
        ]
        
        data = []
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info and 'regularMarketPrice' in info:
                data.append({
                    'Symbol': symbol,
                    'Name': info.get('shortName', symbol),
                    'Price': info['regularMarketPrice'],
                    'Change %': info.get('regularMarketChangePercent', 0)
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            # Sort by change percentage
            gainers = df[df['Change %'] > 0].sort_values('Change %', ascending=False)
            losers = df[df['Change %'] < 0].sort_values('Change %')
            
            return gainers, losers
        return None, None
    except Exception as e:
        st.error(f"Error getting top gainers/losers: {str(e)}")
        return None, None

def plot_market_trend(symbol="^NSEI", period="1mo"):
    """Plot market trend for given symbol"""
    try:
        # Get historical data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if not hist.empty:
            # Create figure
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Price'
            ))
            
            # Add volume bars
            fig.add_trace(go.Bar(
                x=hist.index,
                y=hist['Volume'],
                name='Volume',
                yaxis='y2',
                opacity=0.3
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} Price Chart",
                yaxis_title="Price",
                yaxis2=dict(
                    title="Volume",
                    overlaying='y',
                    side='right'
                ),
                template="plotly_dark",
                xaxis_rangeslider_visible=False
            )
            
            return fig
        return None
    except Exception as e:
        st.error(f"Error plotting market trend: {str(e)}")
        return None

def show_market_activity():
    """Display the market activity page"""
    st.title("📈 Market Activity")
    
    # Market Status
    market_status = get_market_status()
    status_color = "green" if market_status == "Open" else "red"
    st.markdown(f"### Market Status: <span style='color:{status_color}'>{market_status}</span>", unsafe_allow_html=True)
    
    # Market Overview
    st.header("🌍 Market Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nifty = yf.Ticker("^NSEI")
        nifty_info = nifty.info
        if nifty_info and 'regularMarketPrice' in nifty_info:
            st.metric(
                "NIFTY 50",
                f"₹{nifty_info['regularMarketPrice']:,.2f}",
                f"{nifty_info.get('regularMarketChangePercent', 0):+.2f}%"
            )
    
    with col2:
        sensex = yf.Ticker("^BSESN")
        sensex_info = sensex.info
        if sensex_info and 'regularMarketPrice' in sensex_info:
            st.metric(
                "SENSEX",
                f"₹{sensex_info['regularMarketPrice']:,.2f}",
                f"{sensex_info.get('regularMarketChangePercent', 0):+.2f}%"
            )
    
    with col3:
        market_breadth = get_market_breadth()
        if market_breadth:
            st.metric(
                "Market Breadth",
                f"{market_breadth['ratio']:.2f}",
                f"Advances: {market_breadth['advances']} | Declines: {market_breadth['declines']}"
            )
    
    # Market Trend
    st.header("📊 Market Trend")
    fig = plot_market_trend()
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Sector Performance
    st.header("🏢 Sector Performance")
    sector_data = get_sector_performance()
    if sector_data is not None and not sector_data.empty:
        st.dataframe(
            sector_data,
            column_config={
                "Sector": "Sector",
                "Price": st.column_config.NumberColumn("Price (₹)", format="₹%.2f"),
                "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%")
            },
            hide_index=True,
            use_container_width=True
        )
    
    # Top Gainers and Losers
    st.header("📈 Top Gainers & Losers")
    gainers, losers = get_top_gainers_losers()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Gainers")
        if gainers is not None and not gainers.empty:
            st.dataframe(
                gainers,
                column_config={
                    "Symbol": "Symbol",
                    "Name": "Name",
                    "Price": st.column_config.NumberColumn("Price (₹)", format="₹%.2f"),
                    "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%")
                },
                hide_index=True,
                use_container_width=True
            )
    
    with col2:
        st.subheader("Top Losers")
        if losers is not None and not losers.empty:
            st.dataframe(
                losers,
                column_config={
                    "Symbol": "Symbol",
                    "Name": "Name",
                    "Price": st.column_config.NumberColumn("Price (₹)", format="₹%.2f"),
                    "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%")
                },
                hide_index=True,
                use_container_width=True
            )

if __name__ == "__main__":
    show_market_activity() 