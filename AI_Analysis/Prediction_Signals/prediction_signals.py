import streamlit as st
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import ta
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from scipy import stats
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.preprocessing import MinMaxScaler
import os
from .lstm_model import predict_future_prices

logger = logging.getLogger(__name__)

def calculate_indicators(data, timeframe):
    """Calculate technical indicators with timeframe-specific parameters"""
    if timeframe == '1d':
        # Daily parameters
        ma_period = 20
        ema_period = 50
        rsi_period = 14
        bb_period = 20
        atr_period = 14
        vwap_period = 20
    elif timeframe == '1wk':
        # Weekly parameters
        ma_period = 13  # ~3 months
        ema_period = 26  # ~6 months
        rsi_period = 9
        bb_period = 13
        atr_period = 13
        vwap_period = 13
    else:  # Monthly
        # Monthly parameters
        ma_period = 6   # ~6 months
        ema_period = 12  # ~1 year
        rsi_period = 6
        bb_period = 6
        atr_period = 6
        vwap_period = 6
    
    # Moving Averages
    data['SMA'] = SMAIndicator(data['Close'], window=ma_period).sma_indicator()
    data['EMA'] = EMAIndicator(data['Close'], window=ema_period).ema_indicator()
    
    # MACD with timeframe-specific parameters
    macd_fast = max(5, int(ema_period/4))
    macd_slow = max(10, int(ema_period/2))
    macd_signal = max(5, int(ema_period/6))
    macd = MACD(data['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    
    # RSI
    data['RSI'] = RSIIndicator(data['Close'], window=rsi_period).rsi()
    
    # Bollinger Bands
    bb = BollingerBands(data['Close'], window=bb_period, window_dev=2)
    data['BB_Upper'] = bb.bollinger_hband()
    data['BB_Mid'] = bb.bollinger_mavg()
    data['BB_Lower'] = bb.bollinger_lband()
    
    # Volatility Stop (ATR based)
    atr = AverageTrueRange(data['High'], data['Low'], data['Close'], window=atr_period)
    data['ATR'] = atr.average_true_range()
    data['VStop_Long'] = data['High'].rolling(atr_period).max() - 3 * data['ATR']
    data['VStop_Short'] = data['Low'].rolling(atr_period).min() + 3 * data['ATR']
    
    # Volume Weighted Average Price
    data['VWAP'] = VolumeWeightedAveragePrice(
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        volume=data['Volume'],
        window=vwap_period
    ).volume_weighted_average_price()
    
    return data

def calculate_fibonacci_levels(data):
    """Calculate Fibonacci retracement levels based on recent swing highs and lows"""
    # Find the most recent significant swing high and low
    lookback_period = min(60, len(data))  # Use up to 60 periods or available data
    
    # Find swing high (highest high in lookback period)
    swing_high_idx = data['High'].rolling(window=lookback_period).apply(lambda x: x.argmax(), raw=True).iloc[-1]
    swing_high = data.iloc[int(swing_high_idx)]['High']
    swing_high_date = data.index[int(swing_high_idx)]
    
    # Find swing low (lowest low in lookback period)
    swing_low_idx = data['Low'].rolling(window=lookback_period).apply(lambda x: x.argmin(), raw=True).iloc[-1]
    swing_low = data.iloc[int(swing_low_idx)]['Low']
    swing_low_date = data.index[int(swing_low_idx)]
    
    # Determine if we're in an uptrend or downtrend
    if swing_high_date > swing_low_date:
        # Uptrend (swing high occurred after swing low)
        trend = 'uptrend'
        fib_range = swing_high - swing_low
        price_range = swing_high - swing_low
        base_price = swing_low
    else:
        # Downtrend (swing low occurred after swing high)
        trend = 'downtrend'
        fib_range = swing_high - swing_low
        price_range = swing_high - swing_low
        base_price = swing_high
    
    # Calculate Fibonacci levels
    fib_levels = {
        '0.0': swing_high if trend == 'downtrend' else swing_low,
        '0.236': base_price + (0.236 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '0.382': base_price + (0.382 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '0.5': base_price + (0.5 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '0.618': base_price + (0.618 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '0.786': base_price + (0.786 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '1.0': swing_low if trend == 'downtrend' else swing_high,
        '1.272': base_price + (1.272 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '1.414': base_price + (1.414 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '1.618': base_price + (1.618 * fib_range * (-1 if trend == 'downtrend' else 1)),
    }
    
    return fib_levels, trend, swing_high, swing_low, swing_high_date, swing_low_date

def generate_signals(data, timeframe):
    """Generate buy/sell signals with timeframe-specific thresholds"""
    if timeframe == '1d':
        rsi_buy = 50
        rsi_sell = 50
        min_vol_multiplier = 1.2
    elif timeframe == '1wk':
        rsi_buy = 45
        rsi_sell = 55
        min_vol_multiplier = 1.5
    else:  # Monthly
        rsi_buy = 40
        rsi_sell = 60
        min_vol_multiplier = 2.0
    
    data['Signal'] = ""
    
    # Volume filter (higher volume than average)
    vol_avg = data['Volume'].rolling(20).mean()
    volume_ok = data['Volume'] > (vol_avg * min_vol_multiplier)
    
    # Buy Signals (multiple confirmations)
    buy_conditions = (
        (data['Close'] > data['EMA']) & 
        (data['Close'] > data['VWAP']) & 
        (data['RSI'] > rsi_buy) & 
        (data['MACD'] > data['MACD_Signal']) &
        (data['Close'] > data['VStop_Long']) &
        volume_ok
    )
    
    # Sell Signals (multiple confirmations)
    sell_conditions = (
        (data['Close'] < data['EMA']) & 
        (data['Close'] < data['VWAP']) & 
        (data['RSI'] < rsi_sell) & 
        (data['MACD'] < data['MACD_Signal']) &
        (data['Close'] < data['VStop_Short']) &
        volume_ok
    )
    
    data.loc[buy_conditions, 'Signal'] = "BUY"
    data.loc[sell_conditions, 'Signal'] = "SELL"
    
    return data

def calculate_regression_trend(data, future_periods=5):
    """Calculate linear regression trend line with future projection"""
    x = np.arange(len(data))
    y = data['Close'].values
    
    # Calculate linear regression
    slope, intercept, _, _, _ = stats.linregress(x, y)
    
    # Create trend line
    data['Trend_Line'] = intercept + slope * x
    
    # Future projection
    future_x = np.arange(len(data), len(data) + future_periods)
    future_y = intercept + slope * future_x
    
    # Combine with original data
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_periods, freq='D')
    
    future_data = pd.DataFrame({
        'Trend_Line': future_y,
        'is_future': True
    }, index=future_dates)
    
    data['is_future'] = False
    extended_data = pd.concat([data, future_data])
    
    return extended_data, slope

def plot_enhanced_chart(data, fib_levels, trend, swing_high, swing_low, swing_high_date, swing_low_date, slope):
    """Create an enhanced chart with all indicators and signals"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       row_heights=[0.6, 0.2, 0.2])

    # Split into actual and future data
    actual_data = data[~data['is_future']]
    future_data = data[data['is_future']]

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=actual_data.index,
                                open=actual_data['Open'],
                                high=actual_data['High'],
                                low=actual_data['Low'],
                                close=actual_data['Close'],
                                name='Price'), row=1, col=1)

    # Add indicators
    fig.add_trace(go.Scatter(x=actual_data.index, y=actual_data['SMA'],
                            name='SMA', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=actual_data.index, y=actual_data['EMA'],
                            name='EMA', line=dict(color='purple')), row=1, col=1)
    fig.add_trace(go.Scatter(x=actual_data.index, y=actual_data['VWAP'],
                            name='VWAP', line=dict(color='cyan')), row=1, col=1)

    # Add trend line
    fig.add_trace(go.Scatter(x=data.index, y=data['Trend_Line'],
                            name='Trend Line', line=dict(color='black', dash='dash')), row=1, col=1)

    # Add Fibonacci levels
    for level, price in fib_levels.items():
        if level in ['0.0', '1.0']:
            # Main levels
            fig.add_hline(y=price, line_dash="solid", line_color="purple",
                         annotation_text=f"Fib {level}%", row=1, col=1)
        elif level in ['0.236', '0.382', '0.5', '0.618', '0.786']:
            # Retracement levels
            fig.add_hline(y=price, line_dash="dash", line_color="blue",
                         annotation_text=f"Fib {level}%", row=1, col=1)
        else:
            # Extension levels
            fig.add_hline(y=price, line_dash="dot", line_color="green",
                         annotation_text=f"Fib {level}%", row=1, col=1)

    # Add signals
    buy_signals = actual_data[actual_data['Signal'] == "BUY"]
    sell_signals = actual_data[actual_data['Signal'] == "SELL"]
    
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'] * 0.98,
                            mode='markers', name='BUY', marker=dict(color='green', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'] * 1.02,
                            mode='markers', name='SELL', marker=dict(color='red', size=10)), row=1, col=1)

    # Add MACD
    fig.add_trace(go.Scatter(x=actual_data.index, y=actual_data['MACD'],
                            name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=actual_data.index, y=actual_data['MACD_Signal'],
                            name='Signal', line=dict(color='red')), row=2, col=1)

    # Add RSI
    fig.add_trace(go.Scatter(x=actual_data.index, y=actual_data['RSI'],
                            name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # Update layout
    fig.update_layout(
        title='Enhanced Trading Analysis with Fibonacci Levels',
        yaxis_title='Price (₹)',
        yaxis2_title='MACD',
        yaxis3_title='RSI',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=1000
    )

    return fig

def detect_anomalies(data):
    """Detect price and volume anomalies"""
    anomalies = []
    
    # Volume anomalies
    volume_mean = data['Volume'].mean()
    volume_std = data['Volume'].std()
    volume_zscore = (data['Volume'] - volume_mean) / volume_std
    
    # Price anomalies
    price_mean = data['Close'].mean()
    price_std = data['Close'].std()
    price_zscore = (data['Close'] - price_mean) / price_std
    
    # Detect anomalies (z-score > 2)
    for idx in data.index:
        if abs(volume_zscore[idx]) > 2:
            anomalies.append({
                'date': idx,
                'type': 'Volume spike',
                'value': data['Volume'][idx],
                'zscore': volume_zscore[idx]
            })
        if abs(price_zscore[idx]) > 2:
            anomalies.append({
                'date': idx,
                'type': 'Unusual price movement',
                'value': data['Close'][idx],
                'zscore': price_zscore[idx]
            })
    
    return sorted(anomalies, key=lambda x: x['date'], reverse=True)[:5]

def detect_patterns(data):
    """Detect technical patterns"""
    patterns = {
        'double_bottom': {'detected': False, 'confidence': 0.0},
        'head_and_shoulders': {'detected': False, 'confidence': 0.0}
    }
    
    # Double Bottom detection
    lookback = min(60, len(data))
    local_minima = []
    
    for i in range(1, lookback-1):
        if (data['Low'].iloc[i] < data['Low'].iloc[i-1] and 
            data['Low'].iloc[i] < data['Low'].iloc[i+1]):
            local_minima.append(i)
    
    if len(local_minima) >= 2:
        # Check if last two minima form a double bottom
        last_min = local_minima[-1]
        prev_min = local_minima[-2]
        
        if (abs(data['Low'].iloc[last_min] - data['Low'].iloc[prev_min]) / 
            data['Low'].iloc[prev_min] < 0.02):  # Within 2% of each other
            patterns['double_bottom']['detected'] = True
            patterns['double_bottom']['confidence'] = 0.85
    
    # Head and Shoulders detection
    local_maxima = []
    for i in range(1, lookback-1):
        if (data['High'].iloc[i] > data['High'].iloc[i-1] and 
            data['High'].iloc[i] > data['High'].iloc[i+1]):
            local_maxima.append(i)
    
    if len(local_maxima) >= 3:
        # Check if last three maxima form a head and shoulders
        right_shoulder = local_maxima[-1]
        head = local_maxima[-2]
        left_shoulder = local_maxima[-3]
        
        if (data['High'].iloc[head] > data['High'].iloc[left_shoulder] and 
            data['High'].iloc[head] > data['High'].iloc[right_shoulder] and
            abs(data['High'].iloc[left_shoulder] - data['High'].iloc[right_shoulder]) / 
            data['High'].iloc[left_shoulder] < 0.02):
            patterns['head_and_shoulders']['detected'] = True
            patterns['head_and_shoulders']['confidence'] = 0.80
    
    return patterns

def display_prediction_signals(data, chart_type):
    try:
        # Check if we have enough data
        if len(data) < 63:  # 63 trading days in 3 months
            st.warning("Insufficient data for accurate predictions. Need at least 3 months of historical data.")
            return
        
        # Calculate all indicators
        data = calculate_indicators(data, '1d')
        
        # Calculate Fibonacci levels
        fib_levels, trend, swing_high, swing_low, swing_high_date, swing_low_date = calculate_fibonacci_levels(data)
        
        # Generate trading signals
        data = generate_signals(data, '1d')
        
        # Calculate regression trend
        extended_data, slope = calculate_regression_trend(data)
        
        # Get future predictions
        future_predictions, confidence_scores, future_signals = predict_future_prices(data)
        
        # Detect patterns and anomalies
        patterns = detect_patterns(data)
        anomalies = detect_anomalies(data)
        
        # Create tabs for different chart views
        tab1, tab2, tab3 = st.tabs(["Price", "MACD", "RSI"])
        
        with tab1:
            # Price chart with all indicators
            fig_price = make_subplots(rows=1, cols=1, shared_xaxes=True)
            
            # Add candlestick chart
            fig_price.add_trace(go.Candlestick(x=extended_data.index,
                                             open=extended_data['Open'],
                                             high=extended_data['High'],
                                             low=extended_data['Low'],
                                             close=extended_data['Close'],
                                             name='Price'))
            
            # Add future predictions if available
            if future_predictions is not None:
                future_dates = pd.date_range(start=extended_data.index[-1] + pd.Timedelta(days=1), 
                                           periods=len(future_predictions), freq='D')
                fig_price.add_trace(go.Scatter(x=future_dates, y=future_predictions,
                                             name='Predicted Price', line=dict(color='orange', dash='dot')))
                
                # Add future signals
                for i, (date, price, signal) in enumerate(zip(future_dates, future_predictions, future_signals)):
                    if signal == "BUY":
                        fig_price.add_trace(go.Scatter(x=[date], y=[price * 0.98],
                                                     mode='markers', name='Future BUY',
                                                     marker=dict(color='green', size=10)))
                    elif signal == "SELL":
                        fig_price.add_trace(go.Scatter(x=[date], y=[price * 1.02],
                                                     mode='markers', name='Future SELL',
                                                     marker=dict(color='red', size=10)))
            
            # Add indicators
            fig_price.add_trace(go.Scatter(x=extended_data.index, y=extended_data['SMA'],
                                         name='SMA', line=dict(color='blue')))
            fig_price.add_trace(go.Scatter(x=extended_data.index, y=extended_data['EMA'],
                                         name='EMA', line=dict(color='purple')))
            fig_price.add_trace(go.Scatter(x=extended_data.index, y=extended_data['VWAP'],
                                         name='VWAP', line=dict(color='cyan')))
            
            # Add trend line
            fig_price.add_trace(go.Scatter(x=extended_data.index, y=extended_data['Trend_Line'],
                                         name='Trend Line', line=dict(color='black', dash='dash')))
            
            # Add Fibonacci levels
            for level, price in fib_levels.items():
                if level in ['0.0', '1.0']:
                    fig_price.add_hline(y=price, line_dash="solid", line_color="purple",
                                      annotation_text=f"Fib {level}%")
                elif level in ['0.236', '0.382', '0.5', '0.618', '0.786']:
                    fig_price.add_hline(y=price, line_dash="dash", line_color="blue",
                                      annotation_text=f"Fib {level}%")
                else:
                    fig_price.add_hline(y=price, line_dash="dot", line_color="green",
                                      annotation_text=f"Fib {level}%")
            
            # Add signals
            buy_signals = extended_data[extended_data['Signal'] == "BUY"]
            sell_signals = extended_data[extended_data['Signal'] == "SELL"]
            
            fig_price.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'] * 0.98,
                                         mode='markers', name='BUY', marker=dict(color='green', size=10)))
            fig_price.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'] * 1.02,
                                         mode='markers', name='SELL', marker=dict(color='red', size=10)))
            
            fig_price.update_layout(
                title='Price Analysis with Indicators',
                yaxis_title='Price (₹)',
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                height=600
            )
            
            st.plotly_chart(fig_price, use_container_width=True)
        
        with tab2:
            # MACD chart
            fig_macd = make_subplots(rows=1, cols=1, shared_xaxes=True)
            
            fig_macd.add_trace(go.Scatter(x=extended_data.index, y=extended_data['MACD'],
                                        name='MACD', line=dict(color='blue')))
            fig_macd.add_trace(go.Scatter(x=extended_data.index, y=extended_data['MACD_Signal'],
                                        name='Signal', line=dict(color='red')))
            
            # Color the area between MACD and Signal line
            fig_macd.add_trace(go.Scatter(
                x=extended_data.index,
                y=extended_data['MACD'],
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False
            ))
            
            fig_macd.update_layout(
                title='MACD Analysis',
                yaxis_title='MACD',
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                height=600
            )
            
            st.plotly_chart(fig_macd, use_container_width=True)
        
        with tab3:
            # RSI chart
            fig_rsi = make_subplots(rows=1, cols=1, shared_xaxes=True)
            
            fig_rsi.add_trace(go.Scatter(x=extended_data.index, y=extended_data['RSI'],
                                       name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            
            fig_rsi.update_layout(
                title='RSI Analysis',
                yaxis_title='RSI',
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                height=600
            )
            
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Analysis Summary below the charts
        st.subheader("Analysis Summary")
        
        # Create columns for metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Current price and nearest Fibonacci level
            current_price = data['Close'].iloc[-1]
            nearest_level = min(fib_levels.items(), key=lambda x: abs(x[1] - current_price))
            st.metric("Current Price", format_price(current_price))
            st.metric("Nearest Fibonacci Level", 
                     f"{nearest_level[0]} ({format_price(nearest_level[1])})")
            
            # Trend Analysis
            st.subheader("Trend Analysis")
            st.write(f"Current Trend: {trend.capitalize()}")
            st.write(f"Trend Slope: {slope:.4f}")
            st.write(f"Swing High: {format_price(swing_high)} on {swing_high_date.strftime('%Y-%m-%d')}")
            st.write(f"Swing Low: {format_price(swing_low)} on {swing_low_date.strftime('%Y-%m-%d')}")
            
            # Fibonacci Levels
            st.subheader("Fibonacci Levels")
            for level, price in sorted(fib_levels.items(), key=lambda x: float(x[0])):
                st.write(f"{level}%: {format_price(price)}")
        
        with col2:
            # ML Prediction Summary
            st.subheader("ML Prediction Summary")
            if future_predictions is not None:
                for i in range(min(5, len(future_predictions))):
                    date = (data.index[-1] + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d')
                    pred_price = future_predictions[i]
                    conf_score = confidence_scores[i]
                    price_change = pred_price - current_price
                    price_change_pct = (price_change / current_price) * 100
                    signal = future_signals[i]
                    
                    st.write(f"Date: {date}")
                    st.write(f"Predicted Price: {format_price(pred_price)}")
                    st.write(f"Confidence: {conf_score*100:.1f}%")
                    st.write(f"Price Change: {format_price(price_change)} ({price_change_pct:.2f}%)")
                    st.write(f"Signal: {signal}")
                    st.write("---")
            else:
                st.warning("Unable to generate predictions. Please ensure you have at least 3 months of historical data.")
            
            # Pattern Detection
            st.subheader("Pattern Detection")
            st.write("Double Bottom")
            st.write(f"{'Detected' if patterns['double_bottom']['detected'] else 'Not Detected'} ({patterns['double_bottom']['confidence']*100:.1f}%)")
            st.write("Head and Shoulders")
            st.write(f"{'Detected' if patterns['head_and_shoulders']['detected'] else 'Not Detected'} ({patterns['head_and_shoulders']['confidence']*100:.1f}%)")
            
            # Buy/Sell Signals
            st.subheader("Buy/Sell Signals")
            st.write("RSI")
            st.write(f"{'Buy' if data['RSI'].iloc[-1] < 30 else 'Sell' if data['RSI'].iloc[-1] > 70 else 'Hold'} ({data['RSI'].iloc[-1]:.1f})")
            st.write("MACD")
            st.write(f"{'Buy' if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] else 'Sell'} ({data['MACD'].iloc[-1]:.1f})")
            
            # Anomalies
            st.subheader("Anomalies")
            for anomaly in anomalies:
                st.write(f"{anomaly['date'].strftime('%Y-%m-%d')}")
                st.write(f"{anomaly['type']}")
                st.write(f"{format_price(anomaly['value'])}")
                
    except Exception as e:
        logger.error(f"Error in prediction and signals analysis: {str(e)}")
        st.error(f"Error in prediction and signals analysis: {str(e)}")

def format_price(price):
    return f"₹{price:,.2f}" 