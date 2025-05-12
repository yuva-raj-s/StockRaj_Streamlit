import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SectorAnalysis:
    def __init__(self):
        self.sectors = {
            "NIFTY AUTO": "^CNXAUTO",
            "NIFTY BANK": "^NSEBANK",
            "NIFTY FMCG": "^CNXFMCG",
            "NIFTY IT": "^CNXIT",
            "NIFTY MEDIA": "^CNXMEDIA",
            "NIFTY METAL": "^CNXMETAL",
            "NIFTY PHARMA": "^CNXPHARMA",
            "NIFTY PSU BANK": "^CNXPSUBANK",
            "NIFTY REALTY": "^CNXREALTY"
        }
        
        # Major companies in each sector with verified symbols
        self.sector_companies = {
            "NIFTY AUTO": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJFINANCE.NS", "HEROMOTOCO.NS"],
            "NIFTY BANK": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
            "NIFTY FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
            "NIFTY IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
            "NIFTY MEDIA": ["ZEEL.NS", "SUNTV.NS", "NETWORK18.NS", "DISHTV.NS", "PVRINOX.NS"],
            "NIFTY METAL": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "SAIL.NS", "COALINDIA.NS"],
            "NIFTY PHARMA": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "LUPIN.NS", "AUROPHARMA.NS"],
            "NIFTY PSU BANK": ["SBIN.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "UNIONBANK.NS"],
            "NIFTY REALTY": ["DLF.NS", "GODREJPROP.NS", "PRESTIGE.NS", "OBEROIRLTY.NS", "SUNTV.NS"]
        }

    def get_sector_performance(self, timeframe="daily"):
        """Get performance metrics for all sectors"""
        try:
            performance = {}
            for sector_name, symbol in self.sectors.items():
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1d")
                    info = ticker.info
                    
                    if not data.empty:
                        current_price = data["Close"].iloc[-1]
                        previous_close = data["Open"].iloc[0]
                        change_percent = ((current_price - previous_close) / previous_close) * 100
                        
                        performance[sector_name] = {
                            "current_price": current_price,
                            "change_percent": change_percent,
                            "volume": data["Volume"].iloc[-1],
                            "market_cap": info.get("marketCap", 0),
                            "52w_high": info.get("fiftyTwoWeekHigh", 0),
                            "52w_low": info.get("fiftyTwoWeekLow", 0)
                        }
                except Exception as e:
                    logger.error(f"Error getting data for {sector_name}: {e}")
                    continue
            
            return performance
        except Exception as e:
            logger.error(f"Error getting sector performance: {e}")
            return None

    def plot_sector_trends(self):
        """Plot sector performance trends"""
        try:
            fig = go.Figure()
            
            # Get data for the last 5 days to show trends
            for sector_name, symbol in self.sectors.items():
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="5d")  # Changed from 1d to 5d
                    
                    if not data.empty:
                        # Calculate percentage change from start
                        start_price = data["Close"].iloc[0]
                        data["Change"] = ((data["Close"] - start_price) / start_price) * 100
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data["Change"],
                            name=sector_name,
                            mode='lines+markers',  # Added markers
                            line=dict(width=2),
                            hovertemplate="<b>%{x}</b><br>" +
                                        "Change: %{y:.2f}%<br>" +
                                        f"{sector_name}<extra></extra>"
                        ))
                except Exception as e:
                    logger.error(f"Error plotting trend for {sector_name}: {e}")
                    continue
            
            # Update layout for better visualization
            fig.update_layout(
                title="Sector Performance Trends (5 Days)",
                xaxis_title="Date",
                yaxis_title="Change (%)",
                hovermode="x unified",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template="plotly_white",
                height=500,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Add a horizontal line at y=0
            fig.add_shape(
                type="line",
                x0=fig.data[0].x[0],
                y0=0,
                x1=fig.data[0].x[-1],
                y1=0,
                line=dict(
                    color="gray",
                    width=1,
                    dash="dash",
                )
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error plotting sector trends: {e}")
            return None

    def get_sector_specific_indices(self, sector_name):
        """Get detailed analysis for a specific sector"""
        try:
            if sector_name not in self.sectors:
                logger.error(f"Invalid sector name: {sector_name}")
                return None
            
            symbol = self.sectors[sector_name]
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            info = ticker.info
            
            if data.empty:
                return None
            
            current_price = data["Close"].iloc[-1]
            previous_close = data["Open"].iloc[0]
            change_percent = ((current_price - previous_close) / previous_close) * 100
            
            # Calculate total market cap for the sector
            total_market_cap = 0
            top_companies = []
            for company_symbol in self.sector_companies.get(sector_name, []):
                try:
                    company = yf.Ticker(company_symbol)
                    company_data = company.history(period="1d")
                    company_info = company.info
                    
                    if not company_data.empty:
                        company_price = company_data["Close"].iloc[-1]
                        company_prev_close = company_data["Open"].iloc[0]
                        company_change = ((company_price - company_prev_close) / company_prev_close) * 100
                        
                        # Get market cap in crores (divide by 10^7)
                        market_cap = company_info.get("marketCap", 0) / 10000000
                        total_market_cap += market_cap
                        
                        top_companies.append({
                            "symbol": company_symbol,
                            "name": company_info.get("longName", company_symbol),
                            "price": company_price,
                            "change_percent": company_change,
                            "market_cap": market_cap  # Market cap in crores
                        })
                except Exception as e:
                    logger.error(f"Error getting data for {company_symbol}: {e}")
                    continue
            
            return {
                "current_price": current_price,
                "change_percent": change_percent,
                "volume": data["Volume"].iloc[-1],
                "market_cap": total_market_cap,  # Total market cap in crores
                "52w_high": info.get("fiftyTwoWeekHigh", 0),
                "52w_low": info.get("fiftyTwoWeekLow", 0),
                "top_companies": sorted(top_companies, key=lambda x: x["market_cap"], reverse=True)[:5],
                "analysis": {
                    "overview": f"Analysis of {sector_name} sector performance",
                    "metrics": {
                        "Market Cap": f"₹{total_market_cap:,.2f} Cr",
                        "Volume": f"{data['Volume'].iloc[-1]:,}",
                        "52W High": f"₹{info.get('fiftyTwoWeekHigh', 0):.2f}",
                        "52W Low": f"₹{info.get('fiftyTwoWeekLow', 0):.2f}"
                    },
                    "recommendations": "Based on current market conditions and sector performance"
                }
            }
        except Exception as e:
            logger.error(f"Error getting sector specific indices: {e}")
            return None

    def display(self):
        """Display sector analysis in Streamlit"""
        st.title("Sector Analysis")
        
        # Get sector performance
        performance = self.get_sector_performance()
        if performance:
            st.subheader("Sector Performance")
            cols = st.columns(3)
            for i, (sector, data) in enumerate(performance.items()):
                with cols[i % 3]:
                    st.metric(
                        sector,
                        f"₹{data['current_price']:.2f}",
                        f"{data['change_percent']:.2f}%",
                        delta_color="normal"
                    )
        
        # Plot sector trends
        fig = self.plot_sector_trends()
        if fig:
            st.subheader("Sector Trends")
            st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed sector analysis
        st.subheader("Detailed Sector Analysis")
        selected_sector = st.selectbox("Select Sector", list(self.sectors.keys()))
        analysis = self.get_sector_specific_indices(selected_sector)
        
        if analysis:
            # Display sector metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Current Price",
                    f"₹{analysis['current_price']:.2f}",
                    f"{analysis['change_percent']:.2f}%",
                    delta_color="normal"
                )
                st.metric("Volume", f"{analysis['volume']:,}")
            with col2:
                st.metric("Market Cap", f"₹{analysis['market_cap']:,.2f} Cr")
                st.metric("52W High", f"₹{analysis['52w_high']:.2f}")
                st.metric("52W Low", f"₹{analysis['52w_low']:.2f}")
            
            # Display top companies
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
                    hide_index=True
                )

if __name__ == "__main__":
    # Create and display the sector analysis
    sector_analysis = SectorAnalysis()
    sector_analysis.display() 