import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from Market_Activity.Overview.market_overview import MarketOverview
from Market_Activity.Sector.sector_analysis import SectorAnalysis

# Define market indices
MARKET_INDICES = {
    'NIFTY 50': '^NSEI',
    'SENSEX': '^BSESN',
    'NIFTY BANK': '^NSEBANK',
    'NIFTY NEXT 50': '^NSMIDCP',
    'NIFTY MIDCAP 50': '^NSEMDCP50',
    'NIFTY MIDCAP 100': '^CRSMID',
    'NIFTY SMALLCAP 100': '^CNXSC',
    'NIFTY 100': '^CNX100',
    'NIFTY 200': '^NSE200',
    'NIFTY 500': '^CRSLDX'
}

# Define sector indices
SECTOR_INDICES = {
    'NIFTY IT': '^CNXIT',
    'NIFTY PHARMA': '^CNXPHARMA',
    'NIFTY AUTO': '^CNXAUTO',
    'NIFTY METAL': '^CNXMETAL',
    'NIFTY FMCG': '^CNXFMCG',
    'NIFTY ENERGY': '^CNXENERGY',
    'NIFTY PSU BANK': '^CNXPSUBANK',
    'NIFTY BANK': '^NSEBANK',
    'NIFTY MEDIA': '^CNXMEDIA',
    'NIFTY INFRA': '^CNXINFRA'
}

def get_index_data(symbol):
    """Get data for a specific index"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        info = ticker.info
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Open'].iloc[0]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            return {
                'current': current_price,
                'change': change,
                'change_pct': change_pct,
                'volume': hist['Volume'].iloc[-1],
                'high': hist['High'].iloc[-1],
                'low': hist['Low'].iloc[-1]
            }
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
    return None

def get_market_overview():
    """Get market overview data"""
    overview_data = []
    for name, symbol in MARKET_INDICES.items():
        data = get_index_data(symbol)
        if data:
            overview_data.append({
                'Index': name,
                'Current': f"₹{data['current']:,.2f}",
                'Change': f"₹{data['change']:,.2f}",
                'Change %': f"{data['change_pct']:.2f}%",
                'Volume': f"{data['volume']:,.0f}"
            })
    return pd.DataFrame(overview_data)

def get_market_volatility():
    """Calculate market volatility using VIX"""
    try:
        vix = yf.Ticker("^INDIAVIX")
        hist = vix.history(period="1mo")
        if not hist.empty:
            current_vix = hist['Close'].iloc[-1]
            vix_change = hist['Close'].iloc[-1] - hist['Close'].iloc[0]
            vix_change_pct = (vix_change / hist['Close'].iloc[0]) * 100
            
            return {
                'current': current_vix,
                'change': vix_change,
                'change_pct': vix_change_pct,
                'trend': 'Increasing' if vix_change > 0 else 'Decreasing'
            }
    except Exception as e:
        st.error(f"Error fetching VIX data: {str(e)}")
    return None

def get_market_movers():
    """Get top gainers, losers, 52W high, and 52W low stocks"""
    try:
        # Get NIFTY 50 components
        nifty = yf.Ticker("^NSEI")
        components = nifty.info.get('components', [])
        
        if not components:
            return None
            
        movers_data = []
        for symbol in components:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1d")
                
                if not hist.empty and info:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Open'].iloc[0]
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    movers_data.append({
                        'Symbol': symbol,
                        'Name': info.get('shortName', symbol),
                        'Current Price': current_price,
                        'Change %': change_pct,
                        '52W High': info.get('fiftyTwoWeekHigh', 0),
                        '52W Low': info.get('fiftyTwoWeekLow', 0)
                    })
            except:
                continue
                
        df = pd.DataFrame(movers_data)
        
        # Get top gainers and losers
        top_gainers = df.nlargest(5, 'Change %')
        top_losers = df.nsmallest(5, 'Change %')
        
        # Get 52W high and low stocks
        high_52w = df[df['Current Price'] >= df['52W High'] * 0.99].nlargest(5, 'Current Price')
        low_52w = df[df['Current Price'] <= df['52W Low'] * 1.01].nsmallest(5, 'Current Price')
        
        return {
            'top_gainers': top_gainers,
            'top_losers': top_losers,
            'high_52w': high_52w,
            'low_52w': low_52w
        }
    except Exception as e:
        st.error(f"Error getting market movers: {str(e)}")
    return None

def get_sector_performance():
    """Get performance data for all sectors"""
    sector_data = []
    for name, symbol in SECTOR_INDICES.items():
        data = get_index_data(symbol)
        if data:
            sector_data.append({
                'Sector': name,
                'Current': f"₹{data['current']:,.2f}",
                'Change': f"₹{data['change']:,.2f}",
                'Change %': f"{data['change_pct']:.2f}%",
                'Volume': f"{data['volume']:,.0f}"
            })
    return pd.DataFrame(sector_data)

def plot_sector_trends():
    """Plot sector performance trends"""
    try:
        fig = go.Figure()
        
        for name, symbol in SECTOR_INDICES.items():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            
            if not hist.empty:
                # Calculate normalized prices
                normalized_prices = hist['Close'] / hist['Close'].iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=normalized_prices,
                    name=name,
                    mode='lines'
                ))
        
        fig.update_layout(
            title='Sector Performance Trends (Normalized)',
            xaxis_title='Date',
            yaxis_title='Normalized Price (%)',
            template='plotly_dark',
            height=600
        )
        
        return fig
    except Exception as e:
        st.error(f"Error plotting sector trends: {str(e)}")
    return None

def show_market_activity():
    """Display the Market Activity page"""
    st.title("📊 Market Activity")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Overview", "Sectors"])
    
    with tab1:
        st.header("Market Overview")
        
        # Initialize MarketOverview
        market_overview = MarketOverview()
        
        # Get market summary
        summary = market_overview.get_market_summary()
        
        if summary:
            # Display market pulse
            pulse = summary['market_pulse']
            if pulse:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "NIFTY 50",
                        f"₹{pulse['current_price']:,.2f}",
                        f"{pulse['change_percent']:+.2f}%"
                    )
                with col2:
                    st.metric("Day High", f"₹{pulse['day_high']:,.2f}")
                    st.metric("Day Low", f"₹{pulse['day_low']:,.2f}")
                with col3:
                    st.metric("Volume", f"{pulse['volume']:,.0f}")
                    st.metric("Previous Close", f"₹{pulse['previous_close']:,.2f}")
            
            # Display broad indices
            st.subheader("Broad Indices")
            indices_data = summary['broad_indices']
            if indices_data:
                indices_df = pd.DataFrame([
                    {
                        'Index': name,
                        'Price': f"₹{data['price']:,.2f}",
                        'Change': f"₹{data['change']:,.2f}",
                        'Change %': f"{data['change_percent']:+.2f}%"
                    }
                    for name, data in indices_data.items()
                ])
                st.dataframe(indices_df, use_container_width=True)
            
            # Display market volatility
            st.subheader("Market Volatility")
            volatility = summary['volatility']
            if volatility:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "India VIX",
                        f"{volatility['current_vix']:.2f}",
                        f"{volatility['change']:+.2f}"
                    )
                with col2:
                    st.metric("Change %", f"{volatility['change_percent']:+.2f}%")
                with col3:
                    st.metric("Previous Close", f"{volatility['previous_close']:.2f}")
            
            # Display market movers
            st.header("Market Movers")
            movers = summary['market_movers']
            if movers:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top Gainers")
                    gainers_df = pd.DataFrame(movers['top_gainers'])
                    st.dataframe(
                        gainers_df[['symbol', 'name', 'price', 'change_percent']],
                        column_config={
                            'symbol': 'Symbol',
                            'name': 'Name',
                            'price': st.column_config.NumberColumn('Price (₹)', format='₹%.2f'),
                            'change_percent': st.column_config.NumberColumn('Change %', format='%.2f%%')
                        },
                        use_container_width=True
                    )
                    
                    st.subheader("52W High")
                    high_df = pd.DataFrame(movers['52w_high'])
                    st.dataframe(
                        high_df[['symbol', 'name', 'price', '52w_high']],
                        column_config={
                            'symbol': 'Symbol',
                            'name': 'Name',
                            'price': st.column_config.NumberColumn('Price (₹)', format='₹%.2f'),
                            '52w_high': st.column_config.NumberColumn('52W High (₹)', format='₹%.2f')
                        },
                        use_container_width=True
                    )
                
                with col2:
                    st.subheader("Top Losers")
                    losers_df = pd.DataFrame(movers['top_losers'])
                    st.dataframe(
                        losers_df[['symbol', 'name', 'price', 'change_percent']],
                        column_config={
                            'symbol': 'Symbol',
                            'name': 'Name',
                            'price': st.column_config.NumberColumn('Price (₹)', format='₹%.2f'),
                            'change_percent': st.column_config.NumberColumn('Change %', format='%.2f%%')
                        },
                        use_container_width=True
                    )
                    
                    st.subheader("52W Low")
                    low_df = pd.DataFrame(movers['52w_low'])
                    st.dataframe(
                        low_df[['symbol', 'name', 'price', '52w_low']],
                        column_config={
                            'symbol': 'Symbol',
                            'name': 'Name',
                            'price': st.column_config.NumberColumn('Price (₹)', format='₹%.2f'),
                            '52w_low': st.column_config.NumberColumn('52W Low (₹)', format='₹%.2f')
                        },
                        use_container_width=True
                    )
    
    with tab2:
        st.header("Sector Analysis")
        
        # Initialize SectorAnalysis
        sector_analysis = SectorAnalysis()
        
        # Display sector performance
        st.subheader("Sector Performance")
        performance = sector_analysis.get_sector_performance()
        if performance:
            # Create a grid of metrics for each sector
            cols = st.columns(3)
            for i, (sector, data) in enumerate(performance.items()):
                with cols[i % 3]:
                    st.metric(
                        sector,
                        f"₹{data['current_price']:,.2f}",
                        f"{data['change_percent']:+.2f}%",
                        delta_color="normal"
                    )
        
        # Plot sector trends
        st.subheader("Sector Trends")
        fig = sector_analysis.plot_sector_trends()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed sector analysis
        st.subheader("Detailed Sector Analysis")
        selected_sector = st.selectbox("Select Sector", list(sector_analysis.sectors.keys()))
        analysis = sector_analysis.get_sector_specific_indices(selected_sector)
        
        if analysis:
            # Display sector metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Current Price",
                    f"₹{analysis['current_price']:,.2f}",
                    f"{analysis['change_percent']:+.2f}%",
                    delta_color="normal"
                )
                st.metric("Volume", f"{analysis['volume']:,}")
            with col2:
                st.metric("Market Cap", f"₹{analysis['market_cap']:,.2f} Cr")
                st.metric("52W High", f"₹{analysis['52w_high']:.2f}")
                st.metric("52W Low", f"₹{analysis['52w_low']:.2f}")
            
            # Display top companies in the sector
            st.subheader("Top Companies")
            if analysis['top_companies']:
                df = pd.DataFrame(analysis['top_companies'])
                st.dataframe(
                    df,
                    column_config={
                        "symbol": "Symbol",
                        "name": "Company",
                        "price": st.column_config.NumberColumn("Price (₹)", format="₹%.2f"),
                        "change_percent": st.column_config.NumberColumn("Change %", format="%.2f%%"),
                        "market_cap": st.column_config.NumberColumn("Market Cap (Cr)", format="₹%.2f")
                    },
                    hide_index=True,
                    use_container_width=True
                )

if __name__ == "__main__":
    show_market_activity() 