import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PORTFOLIO_FILE = "portfolio_data.json"
INDIAN_SUFFIX = ".NS"

def normalize_symbol(symbol):
    """Normalize symbol to ensure consistent format with .NS suffix"""
    symbol = symbol.upper().strip()
    if not symbol.endswith(INDIAN_SUFFIX):
        symbol = symbol + INDIAN_SUFFIX
    return symbol

def load_portfolio():
    """Load portfolio data with proper initialization"""
    default_portfolio = {
        "transactions": [],
        "holdings": {},
        "goals": [],
        "notes": {}
    }
    
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.error("Corrupted portfolio file. Creating new one.")
                    data = default_portfolio
                
                # Ensure all required keys exist
                for key in default_portfolio:
                    if key not in data:
                        data[key] = default_portfolio[key]
                
                # Normalize all symbols in holdings
                normalized_holdings = {}
                for symbol, holding_data in data["holdings"].items():
                    normalized_symbol = normalize_symbol(symbol)
                    normalized_holdings[normalized_symbol] = holding_data
                data["holdings"] = normalized_holdings
                return data
        except Exception as e:
            logger.error(f"Error loading portfolio data: {str(e)}")
            # If there's any error, create a new portfolio file
            save_portfolio(default_portfolio)
            return default_portfolio
    else:
        # If file doesn't exist, create it with default data
        save_portfolio(default_portfolio)
        return default_portfolio

def save_portfolio(data):
    """Save portfolio data with error handling"""
    try:
        # Ensure data structure is valid
        for key in ["transactions", "holdings", "goals", "notes"]:
            if key not in data:
                data[key] = []
        
        # Convert any datetime objects to strings
        portfolio_copy = data.copy()
        for transaction in portfolio_copy["transactions"]:
            if isinstance(transaction["date"], datetime):
                transaction["date"] = transaction["date"].strftime("%Y-%m-%d")
        
        # Save to file
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio_copy, f, indent=4)
            
        logger.info("Portfolio data saved successfully")
    except Exception as e:
        logger.error(f"Error saving portfolio data: {str(e)}")
        raise  # Re-raise the exception to handle it in the calling function

def validate_stock_symbol(symbol):
    """Validate if the stock symbol exists and has valid data"""
    try:
        normalized_symbol = normalize_symbol(symbol)
        ticker = yf.Ticker(normalized_symbol)
        
        # Try to get basic info first
        info = ticker.info
        if not info or 'regularMarketPrice' not in info:
            return False, "Invalid stock symbol or no data available"
            
        # Try to get historical data
        hist = ticker.history(period="1d")
        if hist.empty:
            return False, "No historical data available for this symbol"
            
        return True, normalized_symbol
    except Exception as e:
        logger.error(f"Error validating symbol: {str(e)}")
        return False, f"Error validating symbol: {str(e)}"

def add_transaction(symbol, quantity, price, date, transaction_type="BUY", notes=""):
    """Add a new transaction with error handling"""
    try:
        # Validate stock symbol first
        is_valid, result = validate_stock_symbol(symbol)
        if not is_valid:
            st.error(result)
            return
            
        normalized_symbol = result  # Use the validated symbol
        portfolio = load_portfolio()
        
        # Validate input
        if not symbol or not quantity or not price or not date:
            st.error("Please fill in all transaction details")
            return
            
        # Convert date to string if it's a datetime object
        if isinstance(date, datetime):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = str(date)
            
        transaction = {
            "symbol": normalized_symbol,
            "quantity": quantity if transaction_type == "BUY" else -quantity,
            "price": float(price),  # Ensure price is a float
            "date": date_str,
            "type": transaction_type,
            "notes": notes
        }
        
        # Add transaction
        portfolio["transactions"].append(transaction)
        
        # Update holdings
        if normalized_symbol in portfolio["holdings"]:
            current_quantity = portfolio["holdings"][normalized_symbol]["quantity"]
            new_quantity = current_quantity + (quantity if transaction_type == "BUY" else -quantity)
            
            if new_quantity < 0:
                st.error("Cannot sell more shares than owned")
                return
                
            if new_quantity == 0:
                del portfolio["holdings"][normalized_symbol]
            else:
                portfolio["holdings"][normalized_symbol]["quantity"] = new_quantity
                if transaction_type == "BUY":
                    portfolio["holdings"][normalized_symbol]["avg_price"] = (
                        (portfolio["holdings"][normalized_symbol]["avg_price"] * current_quantity + 
                         price * quantity) / new_quantity
                    )
        else:
            if transaction_type == "SELL":
                st.error("Cannot sell shares that are not owned")
                return
            portfolio["holdings"][normalized_symbol] = {
                "quantity": quantity,
                "avg_price": float(price)  # Ensure price is a float
            }
        
        save_portfolio(portfolio)
        st.success(f"Transaction added successfully for {normalized_symbol}")
        st.rerun()  # Force a rerun() to update the display
        
    except Exception as e:
        logger.error(f"Error adding transaction: {str(e)}")
        st.error(f"Error adding transaction: {str(e)}")

def get_stock_data(symbol):
    """Get stock data with improved error handling"""
    try:
        normalized_symbol = normalize_symbol(symbol)
        ticker = yf.Ticker(normalized_symbol)
        
        # Try to get real-time data first
        try:
            info = ticker.info
            if not info or 'regularMarketPrice' not in info:
                return {
                    "symbol": normalized_symbol,
                    "current_price": 0,
                    "fast_info": None,
                    "info": None,
                    "hist": None,
                    "error": "No valid price data available"
                }
            
            current_price = info.get('regularMarketPrice', 0)
            if current_price == 0:
                return {
                    "symbol": normalized_symbol,
                    "current_price": 0,
                    "fast_info": None,
                    "info": None,
                    "hist": None,
                    "error": "No valid price data available"
                }
        except:
            return {
                "symbol": normalized_symbol,
                "current_price": 0,
                "fast_info": None,
                "info": None,
                "hist": None,
                "error": "Error fetching price data"
            }
        
        # Get historical data
        hist = ticker.history(period="6mo", interval="1d")
        if hist.empty:
            return {
                "symbol": normalized_symbol,
                "current_price": current_price,
                "fast_info": None,
                "info": info,
                "hist": None,
                "error": "No historical data available"
            }
        
        return {
            "symbol": normalized_symbol,
            "current_price": current_price,
            "fast_info": None,
            "info": info,
            "hist": hist,
            "error": None
        }
    except Exception as e:
        logger.error(f"Error getting stock data: {str(e)}")
        return {
            "symbol": normalized_symbol,
            "current_price": 0,
            "fast_info": None,
            "info": None,
            "hist": None,
            "error": str(e)
        }

def calculate_portfolio_metrics(portfolio):
    """Calculate portfolio metrics"""
    if not portfolio["holdings"]:
        return None
    
    total_value = 0
    total_invested = 0
    holdings_data = []
    skipped_symbols = []
    
    for symbol, data in portfolio["holdings"].items():
        stock_data = get_stock_data(symbol)
        if stock_data["error"]:
            logger.warning(f"Error fetching data for {symbol}: {stock_data['error']}")
            skipped_symbols.append(symbol)
            continue
        
        current_price = stock_data["current_price"]
        if not current_price or current_price == 0:
            logger.warning(f"No valid price for {symbol}. Skipping in calculations.")
            skipped_symbols.append(symbol)
            continue
        
        quantity = data["quantity"]
        avg_price = data["avg_price"]
        invested = quantity * avg_price
        current_value = quantity * current_price
        pnl = current_value - invested
        
        holdings_data.append({
            "symbol": symbol.replace(INDIAN_SUFFIX, ""),
            "quantity": quantity,
            "avg_price": avg_price,
            "current_price": current_price,
            "invested": invested,
            "current_value": current_value,
            "pnl": pnl,
            "pnl_percent": (pnl / invested) * 100 if invested > 0 else 0
        })
        
        total_value += current_value
        total_invested += invested
    
    if not holdings_data:
        return None
    
    total_pnl = total_value - total_invested
    total_pnl_percent = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
    
    return {
        "total_value": total_value,
        "total_invested": total_invested,
        "total_pnl": total_pnl,
        "total_pnl_percent": total_pnl_percent,
        "holdings": holdings_data,
        "skipped_symbols": skipped_symbols
    }

def calculate_technical_indicators(symbol):
    stock_data = get_stock_data(symbol)
    if stock_data["error"] or stock_data["hist"] is None or stock_data["hist"].empty:
        logger.warning(f"No price history for {symbol}. Technical indicators not available.")
        return {
            "RSI": "N/A",
            "MACD": "N/A",
            "Signal": "N/A",
            "Upper Band": "N/A",
            "Lower Band": "N/A",
            "SMA": "N/A",
            "EMA": "N/A",
            "Volume SMA": "N/A"
        }
    
    hist = stock_data["hist"]
    if len(hist) < 30:
        logger.warning(f"Not enough data for {symbol} to calculate technical indicators.")
        return {
            "RSI": "N/A",
            "MACD": "N/A",
            "Signal": "N/A",
            "Upper Band": "N/A",
            "Lower Band": "N/A",
            "SMA": "N/A",
            "EMA": "N/A",
            "Volume SMA": "N/A"
        }
    
    try:
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma = hist['Close'].rolling(window=20).mean()
        std = hist['Close'].rolling(window=20).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        # EMA
        ema = hist['Close'].ewm(span=20, adjust=False).mean()
        
        # Volume SMA
        volume_sma = hist['Volume'].rolling(window=20).mean()
        
        return {
            "RSI": rsi.dropna().iloc[-1],
            "MACD": macd.dropna().iloc[-1],
            "Signal": signal.dropna().iloc[-1],
            "Upper Band": upper_band.dropna().iloc[-1],
            "Lower Band": lower_band.dropna().iloc[-1],
            "SMA": sma.dropna().iloc[-1],
            "EMA": ema.dropna().iloc[-1],
            "Volume SMA": volume_sma.dropna().iloc[-1]
        }
    except Exception as e:
        logger.error(f"Error calculating technical indicators for {symbol}: {str(e)}")
        return {
            "RSI": "N/A",
            "MACD": "N/A",
            "Signal": "N/A",
            "Upper Band": "N/A",
            "Lower Band": "N/A",
            "SMA": "N/A",
            "EMA": "N/A",
            "Volume SMA": "N/A"
        }

def get_fundamental_metrics(symbol):
    stock_data = get_stock_data(symbol)
    if stock_data["error"] or stock_data["info"] is None:
        return {
            "P/E Ratio": "N/A",
            "ROE": "N/A",
            "Debt/Equity": "N/A",
            "Market Cap": "N/A",
            "Dividend Yield": "N/A",
            "EPS": "N/A",
            "Beta": "N/A",
            "52W High": "N/A",
            "52W Low": "N/A"
        }
    
    info = stock_data["info"]
    return {
        "P/E Ratio": info.get("trailingPE", "N/A"),
        "ROE": info.get("returnOnEquity", "N/A"),
        "Debt/Equity": info.get("debtToEquity", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "Dividend Yield": info.get("dividendYield", "N/A"),
        "EPS": info.get("trailingEps", "N/A"),
        "Beta": info.get("beta", "N/A"),
        "52W High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52W Low": info.get("fiftyTwoWeekLow", "N/A")
    }

def plot_portfolio_performance(portfolio):
    if not portfolio["holdings"]:
        return None
    
    # Get the earliest transaction date
    if not portfolio["transactions"]:
        return None
        
    earliest_date = min(transaction["date"] for transaction in portfolio["transactions"])
    earliest_date = datetime.strptime(earliest_date, "%Y-%m-%d")
    
    symbols = list(portfolio["holdings"].keys())
    try:
        # Download data from the earliest transaction date
        df = yf.download(
            symbols,
            start=earliest_date,
            end=datetime.now(),
            interval="1d",
            group_by='ticker'
        )
        
        if df.empty:
            logger.warning("No historical data available for portfolio performance chart.")
            return None
        
        # Calculate portfolio value over time
        portfolio_value = pd.Series(0, index=df.index)
        for symbol in symbols:
            if symbol in df.columns.levels[0]:  # Check if symbol exists in multi-level columns
                # Try Adj Close first, fall back to Close if not available
                price_data = df[symbol]['Adj Close'] if 'Adj Close' in df[symbol].columns else df[symbol]['Close']
                quantity = portfolio["holdings"][symbol]["quantity"]
                portfolio_value += price_data.fillna(0) * quantity
        
        if portfolio_value.empty or portfolio_value.isnull().all():
            logger.warning("No valid price data for portfolio performance chart.")
            return None
        
        # Calculate daily returns
        daily_returns = portfolio_value.pct_change()
        cumulative_returns = (1 + daily_returns).cumprod()
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add portfolio value line
        fig.add_trace(go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue')
        ))
        
        # Add cumulative returns line
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values * 100,
            mode='lines',
            name='Cumulative Returns (%)',
            line=dict(color='green'),
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title="Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Value (INR)",
            yaxis2=dict(
                title="Cumulative Returns (%)",
                overlaying='y',
                side='right'
            ),
            template="plotly_dark",
            hovermode='x unified'
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error plotting portfolio performance: {str(e)}")
        return None

def remove_transaction(portfolio, transaction_index):
    """Remove a transaction and update holdings"""
    try:
        # Get the transaction to be removed
        transaction = portfolio["transactions"][transaction_index]
        symbol = transaction["symbol"]
        quantity = transaction["quantity"]
        price = transaction["price"]
        
        # Update holdings
        if symbol in portfolio["holdings"]:
            current_quantity = portfolio["holdings"][symbol]["quantity"]
            new_quantity = current_quantity - quantity  # Subtract the transaction quantity
            
            if new_quantity < 0:
                st.error("Cannot remove transaction: Would result in negative holdings")
                return False
                
            if new_quantity == 0:
                del portfolio["holdings"][symbol]
            else:
                portfolio["holdings"][symbol]["quantity"] = new_quantity
                # Recalculate average price
                if quantity > 0:  # If it was a buy transaction
                    portfolio["holdings"][symbol]["avg_price"] = (
                        (portfolio["holdings"][symbol]["avg_price"] * current_quantity - 
                         price * quantity) / new_quantity
                    )
        
        # Remove the transaction
        portfolio["transactions"].pop(transaction_index)
        save_portfolio(portfolio)
        return True
        
    except Exception as e:
        logger.error(f"Error removing transaction: {str(e)}")
        return False

def show_portfolio():
    """Display the portfolio page"""
    st.title("💼 Portfolio")
    
    # Load portfolio data
    portfolio = load_portfolio()
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📝 Transactions", "🎯 Goals"])
    
    with tab1:
        _display_portfolio_overview(portfolio)
    
    with tab2:
        _display_transactions(portfolio)
    
    with tab3:
        _display_goals(portfolio)

def _display_portfolio_overview(portfolio):
    """Display portfolio overview section"""
    st.header("Portfolio Overview")
    
    # Add clear all button
    if st.button("Clear Entire Portfolio", key="clear_portfolio"):
        portfolio["transactions"] = []
        portfolio["holdings"] = {}
        portfolio["goals"] = []
        portfolio["notes"] = {}
        save_portfolio(portfolio)
        st.success("Portfolio cleared!")
        st.rerun()
    
    # Calculate portfolio metrics
    metrics = calculate_portfolio_metrics(portfolio)
    
    if metrics:
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Value",
                f"₹{metrics['total_value']:,.2f}",
                f"{metrics['total_pnl_percent']:.2f}%",
                delta_color="normal"
            )
        with col2:
            st.metric("Total Invested", f"₹{metrics['total_invested']:,.2f}")
        with col3:
            st.metric("Total P&L", f"₹{metrics['total_pnl']:,.2f}")
        with col4:
            st.metric("P&L %", f"{metrics['total_pnl_percent']:.2f}%")
        
        # Display holdings table
        st.header("Holdings")
        holdings_df = pd.DataFrame(metrics['holdings'])
        st.dataframe(
            holdings_df,
            column_config={
                "symbol": "Symbol",
                "quantity": "Quantity",
                "avg_price": st.column_config.NumberColumn("Avg Price", format="₹%.2f"),
                "current_price": st.column_config.NumberColumn("Current Price", format="₹%.2f"),
                "invested": st.column_config.NumberColumn("Invested", format="₹%.2f"),
                "current_value": st.column_config.NumberColumn("Current Value", format="₹%.2f"),
                "pnl": st.column_config.NumberColumn("P&L", format="₹%.2f"),
                "pnl_percent": st.column_config.NumberColumn("P&L %", format="%.2f%%")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Display portfolio performance chart
        st.header("Portfolio Performance")
        fig = plot_portfolio_performance(portfolio)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No holdings in your portfolio. Add some transactions to get started!")

def _display_transactions(portfolio):
    """Display transactions section"""
    st.header("Transactions")
    
    # Add new transaction form
    with st.form("add_transaction"):
        st.subheader("Add New Transaction")
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Symbol")
            quantity = st.number_input("Quantity", min_value=1, step=1)
            price = st.number_input("Price", min_value=0.01, step=0.01)
        
        with col2:
            date = st.date_input("Date")
            transaction_type = st.selectbox("Type", ["BUY", "SELL"])
            notes = st.text_area("Notes")
        
        if st.form_submit_button("Add Transaction"):
            add_transaction(symbol, quantity, price, date, transaction_type, notes)
            st.rerun()
        
    # Display transaction history
    st.subheader("Transaction History")
    if portfolio["transactions"]:
        transactions_df = pd.DataFrame(portfolio["transactions"])
        st.dataframe(
            transactions_df,
            column_config={
                "symbol": "Symbol",
                "quantity": "Quantity",
                "price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                "date": "Date",
                "type": "Type",
                "notes": "Notes"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No transactions recorded yet.")

def _display_goals(portfolio):
    """Display goals section"""
    st.header("Investment Goals")
            
    # Add new goal form
    with st.form("add_goal"):
        st.subheader("Add New Goal")
        col1, col2 = st.columns(2)
        
        with col1:
            goal_name = st.text_input("Goal Name")
            target_amount = st.number_input("Target Amount", min_value=0.01, step=1000.0)
        
        with col2:
            target_date = st.date_input("Target Date")
            priority = st.selectbox("Priority", ["High", "Medium", "Low"])
        
        if st.form_submit_button("Add Goal"):
            portfolio["goals"].append({
                "name": goal_name,
                "target_amount": target_amount,
                "target_date": target_date.strftime("%Y-%m-%d"),
                "priority": priority,
                "status": "In Progress"
            })
            save_portfolio(portfolio)
            st.rerun()
    
    # Display goals
    st.subheader("Your Goals")
    if portfolio["goals"]:
        goals_df = pd.DataFrame(portfolio["goals"])
        st.dataframe(
            goals_df,
            column_config={
                "name": "Goal Name",
                "target_amount": st.column_config.NumberColumn("Target Amount", format="₹%.2f"),
                "target_date": "Target Date",
                "priority": "Priority",
                "status": "Status"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No investment goals set yet.")

if __name__ == "__main__":
    show_portfolio() 
