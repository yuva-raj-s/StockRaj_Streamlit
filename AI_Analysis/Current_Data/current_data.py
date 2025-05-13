import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def get_timeframe_options():
    return {
        "1D": ("1d", "15m"),
        "1W": ("5d", "15m"),
        "1M": ("1mo", "1d"),
        "3M": ("3mo", "1d"),
        "1Y": ("1y", "1d"),
        "All": ("max", "1d")
    }

def plot_simple_chart(data, chart_type="candlestick"):
    """Create a simple price chart based on the selected chart type"""
    try:
        # Create figure with secondary y-axis for volume
        fig = make_subplots(rows=1, cols=1, 
                           specs=[[{"secondary_y": True}]],
                           vertical_spacing=0.03)

        # Add price chart based on type
        if chart_type.lower() == "line":
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Price',
                line=dict(color='#00ff00', width=2)
            ), secondary_y=False)
        elif chart_type.lower() == "area":
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Price',
                fill='tozeroy',
                line=dict(color='#00ff00', width=2)
            ), secondary_y=False)
        elif chart_type.lower() == "baseline":
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Price',
                line=dict(color='#00ff00', width=2)
            ), secondary_y=False)
            fig.add_hline(
                y=data['Close'].iloc[0],
                line_dash="dash",
                line_color="red",
                annotation_text="Baseline"
            )
        else:  # candlestick
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            ), secondary_y=False)

        # Add volume as a bar chart with low opacity
        colors = ['#00ff00' if data['Close'][i] >= data['Open'][i] else '#ff0000' 
                 for i in range(len(data))]
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.2
        ), secondary_y=True)

        # Add technical indicators if available
        if 'SMA_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                name='SMA 20',
                line=dict(color='#ffa500', width=1)
            ), secondary_y=False)

        if 'EMA_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['EMA_20'],
                name='EMA 20',
                line=dict(color='#00ffff', width=1)
            ), secondary_y=False)

        if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_Upper'],
                name='BB Upper',
                line=dict(color='#808080', width=1, dash='dash')
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_Lower'],
                name='BB Lower',
                line=dict(color='#808080', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            ), secondary_y=False)

        # Update layout with modern styling
        fig.update_layout(
            title='Price Chart',
            template='plotly_dark',
            height=800,
            plot_bgcolor='rgba(30,33,48,0.95)',
            paper_bgcolor='rgba(30,33,48,1)',
            font=dict(family='Segoe UI', size=14, color='white'),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            xaxis_rangeslider_visible=False
        )

        # Update y-axes
        fig.update_yaxes(title_text="Price (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Volume", secondary_y=True, showgrid=False)

        # Update x-axes
        fig.update_xaxes(
            type='date',
            tickformat='%H:%M' if pd.Timedelta(data.index[-1] - data.index[0]) < pd.Timedelta(days=1) else '%Y-%m-%d'
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        st.error(f"Error creating chart: {str(e)}")
        return None

def format_price(price):
    return f"₹{price:,.2f}"

def display_current_data(data, chart_type):
    """Display current market data with chart and indicators"""
    try:
        if data is None or data.empty:
            st.error("No data available to display")
            return

        # Add chart type selector
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Candlestick", "Line", "Area", "Baseline"],
            index=0,
            key="current_data_chart_type"
        )

        # Display the chart
        fig = plot_simple_chart(data, chart_type.lower())
        if fig:
            st.plotly_chart(fig, use_container_width=True)

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