import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from watchlist_operations import WatchlistManager
from price_alerts import PriceAlertManager

def show_watchlist():
    """Display the watchlist page"""
    st.title("👀 Watchlist")
    
    # Initialize managers
    watchlist_manager = WatchlistManager()
    alert_manager = PriceAlertManager()
    
    # Create tabs for Watchlist and Price Alerts
    tab1, tab2 = st.tabs(["📊 Watchlist", "🔔 Price Alerts"])
    
    with tab1:
        _display_watchlist(watchlist_manager)
    
    with tab2:
        _display_price_alerts(alert_manager)

def _display_watchlist(watchlist_manager):
    """Display the main watchlist section"""
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Symbol Search and Add
        st.header("🔍 Add Stocks to Watchlist")
        search_query = st.text_input("Search for stocks", key="stock_search")
        
        if search_query:
            search_results = watchlist_manager.search_stocks(search_query)
            if search_results:
                # Create options list with proper error handling
                options = []
                for result in search_results:
                    symbol = result.get('symbol', '')
                    name = result.get('shortname', result.get('name', symbol))
                    options.append(f"{symbol} - {name}")
                
                if options:
                    selected_symbol = st.selectbox(
                        "Select a stock",
                        options=options
                    )
                    if st.button("Add to Watchlist"):
                        symbol = selected_symbol.split(" - ")[0]
                        watchlist_manager.add_to_watchlist(symbol)
                        st.success(f"Added {symbol} to watchlist!")
                        st.experimental_rerun()
            else:
                st.info("No stocks found matching your search.")
        
        # Display Watchlist Table
        st.header("📋 Your Watchlist")
        watchlist_data = watchlist_manager.get_watchlist_data()
        if not watchlist_data.empty:
            # Display the data with enhanced formatting
            st.dataframe(
                watchlist_data,
                use_container_width=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Current Price": st.column_config.TextColumn("Current Price", width="small"),
                    "Change": st.column_config.TextColumn("Change", width="small"),
                    "Change %": st.column_config.TextColumn("Change %", width="small"),
                    "Day High": st.column_config.TextColumn("Day High", width="small"),
                    "Day Low": st.column_config.TextColumn("Day Low", width="small")
                }
            )
            
            # Add remove button for each stock
            for symbol in watchlist_data['Symbol']:
                if st.button(f"Remove {symbol}", key=f"remove_{symbol}"):
                    watchlist_manager.remove_from_watchlist(f"{symbol}.NS")
                    st.experimental_rerun()
        else:
            st.info("Your watchlist is empty. Add some stocks to get started!")
    
    with col2:
        # Display watchlist summary
        st.header("📊 Watchlist Summary")
        if not watchlist_data.empty:
            try:
                # Calculate summary metrics with proper handling of market cap in crores
                total_market_cap = sum(float(x.replace('₹', '').replace(',', '').replace(' Cr', '')) for x in watchlist_data['Market Cap'])
                avg_pe = sum(float(x) for x in watchlist_data['P/E Ratio'] if x != 'N/A') / len(watchlist_data)
                avg_dividend = sum(float(x.replace('%', '')) for x in watchlist_data['Dividend Yield']) / len(watchlist_data)
                
                st.metric("Total Market Cap", f"₹{total_market_cap:,.2f} Cr")
                st.metric("Average P/E Ratio", f"{avg_pe:.2f}")
                st.metric("Average Dividend Yield", f"{avg_dividend:.2f}%")
            except Exception as e:
                st.error(f"Error calculating summary metrics: {str(e)}")

def _display_price_alerts(alert_manager):
    """Display price alerts section"""
    st.header("🔔 Price Alerts")
    
    # Add new alert
    col1, col2, col3 = st.columns(3)
    with col1:
        alert_symbol = st.text_input("Symbol", key="alert_symbol")
    with col2:
        alert_price = st.number_input("Alert Price", min_value=0.0, step=0.01)
    with col3:
        alert_type = st.selectbox("Alert Type", ["Above", "Below"])
    
    if st.button("Add Alert"):
        if alert_symbol and alert_price:
            # Ensure symbol has .NS suffix
            if not alert_symbol.endswith('.NS'):
                alert_symbol = f"{alert_symbol}.NS"
            if alert_manager.add_alert(alert_symbol, alert_price, alert_type):
                st.success("Alert added successfully!")
            else:
                st.warning("This alert already exists!")
    
    # Display active alerts
    st.header("Active Alerts")
    alerts = alert_manager.get_alerts()
    if alerts:
        alert_data = []
        for alert in alerts:
            try:
                symbol = alert['symbol']
                if not symbol.endswith('.NS'):
                    symbol = f"{symbol}.NS"
                
                ticker = yf.Ticker(symbol)
                # Get current price using history
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    created_at = datetime.fromisoformat(alert['created_at']).strftime("%Y-%m-%d %H:%M:%S")
                    
                    alert_data.append({
                        'Symbol': symbol.replace('.NS', ''),
                        'Current Price': f"₹{current_price:.2f}",
                        'Alert Price': f"₹{alert['price']:.2f}",
                        'Type': alert['type'],
                        'Created': created_at
                    })
            except Exception as e:
                st.error(f"Error getting price for {alert['symbol']}: {e}")
                continue
        
        if alert_data:
            st.dataframe(
                pd.DataFrame(alert_data),
                use_container_width=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Current Price": st.column_config.TextColumn("Current Price", width="small"),
                    "Alert Price": st.column_config.TextColumn("Alert Price", width="small"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "Created": st.column_config.TextColumn("Created", width="medium")
                }
            )
            
            # Add remove buttons for alerts
            for alert in alerts:
                symbol = alert['symbol'].replace('.NS', '')
                if st.button(f"Remove Alert for {symbol}", key=f"remove_alert_{alert['symbol']}"):
                    alert_manager.remove_alert(alert['symbol'])
                    st.experimental_rerun()
    else:
        st.info("No active alerts. Add some to get notified!")

if __name__ == "__main__":
    show_watchlist() 