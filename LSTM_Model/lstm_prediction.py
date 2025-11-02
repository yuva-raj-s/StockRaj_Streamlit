import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from scipy import stats

def fetch_stock_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        
        # Adjust period based on interval to ensure enough data points
        if interval == "1mo":
            period = "5y"  # Get more data for monthly analysis
        elif interval == "1wk":
            period = "3y"  # Get more data for weekly analysis
        
        df = stock.history(period=period, interval=interval)
        if df.empty:
            raise ValueError("No data found for this ticker")
            
        # Ensure we have enough data points
        if len(df) < 60:  # Minimum required data points
            raise ValueError(f"Insufficient data points. Got {len(df)}, need at least 60.")
            
        # Handle missing values
        df = df.fillna(method='ffill')  # Forward fill missing values
        df = df.fillna(method='bfill')  # Backward fill any remaining missing values
        
        return df
    except Exception as e:
        raise ValueError(f"Error fetching data: {str(e)}")

def calculate_technical_indicators(df: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
    """Calculate technical indicators with timeframe-specific parameters"""
    try:
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
            # Monthly parameters - adjusted for fewer data points
            ma_period = 6   # ~6 months
            ema_period = 12  # ~1 year
            rsi_period = 6
            bb_period = 6
            atr_period = 6
            vwap_period = 6
            
            # Ensure periods don't exceed available data
            max_period = len(df) // 2
            ma_period = min(ma_period, max_period)
            ema_period = min(ema_period, max_period)
            rsi_period = min(rsi_period, max_period)
            bb_period = min(bb_period, max_period)
            atr_period = min(atr_period, max_period)
            vwap_period = min(vwap_period, max_period)
        
        # Moving Averages
        df['SMA'] = SMAIndicator(df['Close'], window=ma_period).sma_indicator()
        df['EMA'] = EMAIndicator(df['Close'], window=ema_period).ema_indicator()
        
        # MACD with timeframe-specific parameters
        macd_fast = max(5, int(ema_period/4))
        macd_slow = max(10, int(ema_period/2))
        macd_signal = max(5, int(ema_period/6))
        macd = MACD(df['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # RSI
        df['RSI'] = RSIIndicator(df['Close'], window=rsi_period).rsi()
        
        # Bollinger Bands
        bb = BollingerBands(df['Close'], window=bb_period, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Mid'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        
        # Volatility Stop (ATR based)
        atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_period)
        df['ATR'] = atr.average_true_range()
        df['VStop_Long'] = df['High'].rolling(atr_period).max() - 3 * df['ATR']
        df['VStop_Short'] = df['Low'].rolling(atr_period).min() + 3 * df['ATR']
        
        # Volume Weighted Average Price
        df['VWAP'] = VolumeWeightedAveragePrice(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume'],
            window=vwap_period
        ).volume_weighted_average_price()
        
        # Handle any remaining NaN values
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        
        return df
    except Exception as e:
        raise ValueError(f"Error calculating technical indicators: {str(e)}")

def calculate_fibonacci_levels(df: pd.DataFrame):
    """Calculate Fibonacci retracement levels based on recent swing highs and lows"""
    lookback_period = min(60, len(df))
    
    # Find swing high and low
    swing_high_idx = df['High'].rolling(window=lookback_period).apply(lambda x: x.argmax(), raw=True).iloc[-1]
    swing_high = df.iloc[int(swing_high_idx)]['High']
    swing_high_date = df.index[int(swing_high_idx)]
    
    swing_low_idx = df['Low'].rolling(window=lookback_period).apply(lambda x: x.argmin(), raw=True).iloc[-1]
    swing_low = df.iloc[int(swing_low_idx)]['Low']
    swing_low_date = df.index[int(swing_low_idx)]
    
    # Determine trend
    if swing_high_date > swing_low_date:
        trend = 'uptrend'
        fib_range = swing_high - swing_low
        base_price = swing_low
    else:
        trend = 'downtrend'
        fib_range = swing_high - swing_low
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

def detect_anomalies(df: pd.DataFrame, window: int = 20, threshold: float = 2.5) -> pd.DataFrame:
    """Detect price anomalies using multiple indicators and filters"""
    try:
        # Calculate rolling statistics
        df['Rolling_Mean'] = df['Close'].rolling(window=window).mean()
        df['Rolling_Std'] = df['Close'].rolling(window=window).std()
        
        # Calculate z-score
        df['Z_Score'] = (df['Close'] - df['Rolling_Mean']) / df['Rolling_Std']
        
        # Calculate price change percentage
        df['Price_Change_Pct'] = df['Close'].pct_change() * 100
        
        # Calculate volume metrics
        df['Volume_MA'] = df['Volume'].rolling(window=window).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Calculate volatility
        df['Volatility'] = df['Close'].rolling(window=window).std() / df['Close'].rolling(window=window).mean() * 100
        
        # Calculate momentum
        df['Momentum'] = df['Close'].pct_change(periods=window) * 100
        
        # Mark anomalies with multiple conditions
        df['Is_Anomaly'] = (
            (abs(df['Z_Score']) > threshold) &  # Z-score threshold
            (abs(df['Price_Change_Pct']) > 2.0) &  # Minimum price change
            (df['Volume_Ratio'] > 1.5) &  # Volume confirmation
            (abs(df['Z_Score']) > abs(df['Z_Score'].shift(1))) &  # Increasing deviation
            (abs(df['Z_Score']) > abs(df['Z_Score'].shift(-1))) &  # Peak deviation
            (df['Volatility'] > df['Volatility'].rolling(window=window).mean())  # High volatility
        )
        
        # Calculate anomaly strength (0-100)
        df['Anomaly_Strength'] = (
            (abs(df['Z_Score']) / threshold) *  # Z-score component
            (abs(df['Price_Change_Pct']) / 2.0) *  # Price change component
            (df['Volume_Ratio'] / 1.5) *  # Volume component
            (df['Volatility'] / df['Volatility'].rolling(window=window).mean())  # Volatility component
        ) * 25  # Scale to 0-100
        
        # Remove consecutive anomalies (keep only the strongest)
        df['Is_Anomaly'] = df['Is_Anomaly'].astype(bool)
        df['Is_Anomaly'] = df['Is_Anomaly'] & (~df['Is_Anomaly'].shift(1).fillna(False))
        
        # Add anomaly type (Bullish/Bearish)
        df['Anomaly_Type'] = np.where(
            df['Is_Anomaly'],
            np.where(df['Price_Change_Pct'] > 0, 'Bullish', 'Bearish'),
            None
        )
        
        return df
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        return df

def generate_signals(df: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
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
    
    df['Signal'] = 0  # 0: no signal, 1: buy, -1: sell
    
    # Volume filter
    vol_avg = df['Volume'].rolling(20).mean()
    volume_ok = df['Volume'] > (vol_avg * min_vol_multiplier)
    
    # Buy Signals (multiple confirmations)
    buy_conditions = (
        (df['Close'] > df['EMA']) & 
        (df['Close'] > df['VWAP']) & 
        (df['RSI'] > rsi_buy) & 
        (df['MACD'] > df['MACD_Signal']) &
        (df['Close'] > df['VStop_Long']) &
        volume_ok
    )
    
    # Sell Signals (multiple confirmations)
    sell_conditions = (
        (df['Close'] < df['EMA']) & 
        (df['Close'] < df['VWAP']) & 
        (df['RSI'] < rsi_sell) & 
        (df['MACD'] < df['MACD_Signal']) &
        (df['Close'] < df['VStop_Short']) &
        volume_ok
    )
    
    df.loc[buy_conditions, 'Signal'] = 1
    df.loc[sell_conditions, 'Signal'] = -1
    
    return df

def prepare_data(df: pd.DataFrame, sequence_length: int = 30):
    """Prepare data for LSTM model with more features"""
    # Feature engineering
    df['Return'] = df['Close'].pct_change()
    df['DayOfWeek'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else pd.to_datetime(df.index).dayofweek
    df['Month'] = df.index.month if hasattr(df.index, 'month') else pd.to_datetime(df.index).month
    # Add more technical indicators if not already present
    if 'EMA' not in df.columns:
        df['EMA'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    if 'SMA' not in df.columns:
        df['SMA'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    if 'MACD' not in df.columns:
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
    if 'RSI' not in df.columns:
        df['RSI'] = RSIIndicator(df['Close']).rsi()
    # Select features
    features = ['Close', 'Volume', 'RSI', 'MACD', 'EMA', 'SMA', 'Return', 'DayOfWeek', 'Month']
    data = df[features].dropna().values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict close
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def create_lstm_model(sequence_length: int, n_features: int, lstm_units=[32, 32], dropout_rate=0.3):
    """Create and compile enhanced Bidirectional LSTM model"""
    model = Sequential()
    # Input layer
    model.add(Bidirectional(LSTM(lstm_units[0], return_sequences=len(lstm_units) > 1), input_shape=(sequence_length, n_features)))
    model.add(Dropout(dropout_rate))

    # Hidden layers
    for i in range(1, len(lstm_units)):
        return_sequences = i < len(lstm_units) - 1
        model.add(Bidirectional(LSTM(lstm_units[i], return_sequences=return_sequences)))
        model.add(Dropout(dropout_rate))

    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_prices(model, last_sequence, scaler, days_to_predict: int = 20):
    """Predict future prices using all features, only inverse-transforming the 'Close' column."""
    future_predictions = []
    current_sequence = last_sequence.copy()
    n_features = current_sequence.shape[1]
    for _ in range(days_to_predict):
        next_pred = model.predict(current_sequence.reshape(1, -1, n_features), verbose=0)
        # Prepare the next input: shift left, append new prediction for 'Close', keep other features as last
        next_input = current_sequence.copy()
        next_input = np.roll(next_input, -1, axis=0)
        # Replace only the 'Close' value in the last row with the predicted value
        next_input[-1, 0] = next_pred[0, 0]
        current_sequence = next_input
        future_predictions.append(next_pred[0, 0])
    # Inverse transform: create dummy array with predicted closes in the first column, zeros elsewhere
    dummy = np.zeros((len(future_predictions), n_features))
    dummy[:, 0] = future_predictions
    future_predictions = scaler.inverse_transform(dummy)[:, 0]
    return future_predictions

def create_interactive_plot(df, predictions, ticker, fib_levels):
    """Create an interactive plot with price, predictions, and technical indicators"""
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]],
        vertical_spacing=0.03
    )

    # Add candlestick chart with custom colors
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#00ff00',
        decreasing_line_color='#ff0000',
        showlegend=True
    ), secondary_y=False)

    # Add predictions with custom styling
    future_dates = [df.index[-1] + timedelta(days=x+1) for x in range(len(predictions))]
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions.flatten(),
        name='Predicted Price',
        line=dict(color='#00ffff', dash='dot', width=3),
        mode='lines+markers',
        marker=dict(size=8, color='#00ffff', symbol='star'),
        showlegend=True
    ), secondary_y=False)

    # Add volume as a bar chart with low opacity
    colors = ['#00ff00' if df['Close'][i] >= df['Open'][i] else '#ff0000' 
             for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.2,
        showlegend=True
    ), secondary_y=True)

    # Add technical indicators with consistent styling
    if 'SMA' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA'],
            name='SMA',
            line=dict(color='#ffa500', width=1.5),
            showlegend=True
        ), secondary_y=False)

    if 'EMA' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA'],
            name='EMA',
            line=dict(color='#00ffff', width=1.5),
            showlegend=True
        ), secondary_y=False)

    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'],
            name='BB Upper',
            line=dict(color='#808080', dash='dash', width=1),
            showlegend=True
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'],
            name='BB Lower',
            line=dict(color='#808080', dash='dash', width=1),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            showlegend=True
        ), secondary_y=False)

    # Add buy/sell signals with custom icons
    buy_signals = df[(df['Signal'] == 1)]
    sell_signals = df[(df['Signal'] == -1)]
    
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(
                symbol='triangle-up',
                size=16,
                color='#00ff00',
                line=dict(width=2, color='white')
            ),
            showlegend=True,
            hovertemplate='Buy Signal<br>Date: %{x}<br>Price: %{y:.2f}'
        ), secondary_y=False)
    
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(
                symbol='triangle-down',
                size=16,
                color='#ff0000',
                line=dict(width=2, color='white')
            ),
            showlegend=True,
            hovertemplate='Sell Signal<br>Date: %{x}<br>Price: %{y:.2f}'
        ), secondary_y=False)

    # Add anomaly markers
    anomalies = df[df['Is_Anomaly']]
    if not anomalies.empty:
        # Bullish anomalies
        bullish_anomalies = anomalies[anomalies['Anomaly_Type'] == 'Bullish']
        if not bullish_anomalies.empty:
            fig.add_trace(go.Scatter(
                x=bullish_anomalies.index,
                y=bullish_anomalies['Close'],
                mode='markers',
                name='Bullish Anomaly',
                marker=dict(
                    symbol='star',
                    size=20,
                    color='#00ff00',
                    line=dict(width=2, color='white')
                ),
                showlegend=True,
                hovertemplate='Bullish Anomaly<br>Date: %{x}<br>Price: %{y:.2f}<br>Strength: %{customdata:.1f}%',
                customdata=bullish_anomalies['Anomaly_Strength']
            ), secondary_y=False)

        # Bearish anomalies
        bearish_anomalies = anomalies[anomalies['Anomaly_Type'] == 'Bearish']
        if not bearish_anomalies.empty:
            fig.add_trace(go.Scatter(
                x=bearish_anomalies.index,
                y=bearish_anomalies['Close'],
                mode='markers',
                name='Bearish Anomaly',
                marker=dict(
                    symbol='star',
                    size=20,
                    color='#ff0000',
                    line=dict(width=2, color='white')
                ),
                showlegend=True,
                hovertemplate='Bearish Anomaly<br>Date: %{x}<br>Price: %{y:.2f}<br>Strength: %{customdata:.1f}%',
                customdata=bearish_anomalies['Anomaly_Strength']
            ), secondary_y=False)

    # Add Fibonacci levels
    for level, price in fib_levels.items():
        fig.add_hline(
            y=price,
            line_dash="dot",
            line_color="#FFD700",
            annotation_text=f"Fib {level}",
            annotation_position="top right",
            line_width=1
        )

    # Update layout with modern styling
    fig.update_layout(
        title=f'{ticker} Price Prediction & Technicals',
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
            x=1,
            bgcolor='rgba(0,0,0,0)'
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    # Update y-axes
    fig.update_yaxes(
        title_text="Price (₹)",
        secondary_y=False,
        gridcolor='rgba(128,128,128,0.1)',
        zerolinecolor='rgba(128,128,128,0.1)'
    )
    fig.update_yaxes(
        title_text="Volume",
        secondary_y=True,
        showgrid=False,
        zeroline=False
    )

    # Update x-axis
    fig.update_xaxes(
        type='date',
        gridcolor='rgba(128,128,128,0.1)',
        zerolinecolor='rgba(128,128,128,0.1)',
        tickformat='%H:%M' if pd.Timedelta(df.index[-1] - df.index[0]) < pd.Timedelta(days=1) else '%Y-%m-%d'
    )

    return fig

def main():
    # Only set page config if running as standalone app
    if __name__ == '__main__':
        st.set_page_config(
            page_title="LSTM Stock Prediction",
            page_icon="📈",
            layout="wide"
        )
    
    st.title('📊 Stock Price Prediction & Insights')
    st.markdown("""
    Welcome! This app uses advanced AI and technical analysis to help you understand stock trends and make smarter decisions. 
    <span style='color:#FFD700;'>No experience needed!</span> 🚀
    """, unsafe_allow_html=True)

    # Search and Controls Section - 2 columns like tickertape/moneycontrol
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input('Enter Stock Ticker (e.g., RELIANCE):', 'RELIANCE',
                             help="Enter NSE stock symbol (e.g., RELIANCE, TCS, INFY)")
    with col2:
        timeframe = st.selectbox('Timeframe', ['1d', '1wk'], index=0,
                               help="Select the timeframe for analysis")
    with col3:
        prediction_days = st.selectbox('Days to Predict', [5, 10, 15, 20, 30], index=3)
    
    # Additional parameters in a collapsible section
    with st.expander("Advanced Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            anomaly_threshold = st.slider('Anomaly Detection Threshold', 2.0, 4.0, 2.5, 0.1)
        with col2:
            sequence_length = st.selectbox('Sequence Length', [30, 60, 90, 120], index=0)
    
    if st.button('Analyze', type='primary'):
        try:
            with st.spinner('Fetching and analyzing data...'):
                # Fetch and process data
                df = fetch_stock_data(ticker, interval=timeframe)
                df = calculate_technical_indicators(df, timeframe)
                df = detect_anomalies(df, threshold=anomaly_threshold)
                df = generate_signals(df, timeframe)
                fib_levels, trend, swing_high, swing_low, swing_high_date, swing_low_date = calculate_fibonacci_levels(df)
                
                # Stock Chart & Price Analysis Section
                st.markdown("""
                <div style='background-color: #1e2130; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                    <h3 style='color: #ffffff; margin: 0 0 1rem 0;'>Stock Chart & Price Analysis</h3>
                </div>
                """, unsafe_allow_html=True)

                # Price metrics with improved styling
                current_price = df['Close'].iloc[-1]
                open_price = df['Open'].iloc[-1]
                high_price = df['High'].iloc[-1]
                low_price = df['Low'].iloc[-1]
                price_change = current_price - open_price
                price_change_pct = (price_change / open_price) * 100

                cols = st.columns(6)
                metrics = [
                    ('Open', f'₹{open_price:.2f}'),
                    ('High', f'₹{high_price:.2f}'),
                    ('Low', f'₹{low_price:.2f}'),
                    ('Current', f'₹{current_price:.2f}'),
                    ('Change', f'₹{price_change:.2f}'),
                    ('Change %', f'{price_change_pct:+.2f}%')
                ]
                
                for col, (label, value) in zip(cols, metrics):
                    with col:
                        st.metric(label, value, 
                                delta=f"{price_change_pct:+.2f}%" if label == "Change %" else None,
                                delta_color="normal")
                
                # Prepare and train LSTM model
                X, y, scaler = prepare_data(df, sequence_length)
                n_features = X.shape[2]
                model = create_lstm_model(sequence_length, n_features)
                # Train/validation split
                split = int(0.85 * len(X))
                X_train, X_val = X[:split], X[split:]
                y_train, y_val = y[:split], y[split:]
                # Early stopping
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                # Train model with progress bar
                progress_bar = st.progress(0)
                epochs = 50
                for epoch in range(epochs):
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=1, batch_size=8, verbose=0,
                        callbacks=[early_stop]
                    )
                    progress_bar.progress((epoch + 1) / epochs)
                    # Early stopping check
                    if early_stop.stopped_epoch > 0:
                        break
                # Make predictions
                last_sequence = X[-1]  # shape: (sequence_length, n_features)
                predictions = predict_future_prices(model, last_sequence, scaler, prediction_days)
                
                # Create and display interactive plot
                fig = create_interactive_plot(df, predictions, ticker, fib_levels)
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction Metrics Section
                st.markdown("""
                <div style='background-color: #1e2130; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                    <h3 style='color: #ffffff; margin: 0 0 1rem 0;'>Prediction Metrics</h3>
                </div>
                """, unsafe_allow_html=True)

                # Display metrics in a grid
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric('Current Price', f'₹{df["Close"].iloc[-1]:.2f}')
                with col2:
                    st.metric('Predicted Price', f'₹{predictions[-1]:.2f}')
                with col3:
                    price_change = ((predictions[-1] - df["Close"].iloc[-1]) / df["Close"].iloc[-1]) * 100
                    st.metric('Predicted Change', f'{price_change:+.2f}%', 
                            delta=f'{price_change:+.2f}%',
                            delta_color="normal")
                with col4:
                    st.metric('RSI', f'{df["RSI"].iloc[-1]:.2f}')
                
                # Technical Analysis Section
                st.markdown("""
                <div style='background-color: #1e2130; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                    <h3 style='color: #ffffff; margin: 0 0 1rem 0;'>Technical Analysis</h3>
                </div>
                """, unsafe_allow_html=True)

                # Display current signal
                current_signal = df['Signal'].iloc[-1]
                signal_text = "BUY" if current_signal == 1 else "SELL" if current_signal == -1 else "NEUTRAL"
                signal_color = "green" if current_signal == 1 else "red" if current_signal == -1 else "gray"
                
                st.markdown(f"""
                <div style='background-color: #1e2130; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                    <h3 style='color: #ffffff; margin: 0 0 0.5rem 0;'>Trading Signal</h3>
                    <p style='color: {signal_color}; font-size: 1.5rem; font-weight: bold; margin: 0;'>{signal_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display trend and anomaly summary side by side
                col_trend, col_anomaly = st.columns(2)

                with col_trend:
                    st.markdown(f"""
                    <div style='background-color: #1e2130; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
                        <h3 style='color: #ffffff; margin: 0 0 1rem 0; font-size: 1.2rem;'>Trend Analysis</h3>
                        <div style='display: flex; flex-direction: column; gap: 0.5rem;'>
                            <div style='color: #ffffff; font-weight: 500;'>Current Trend: <span style='color: #00ff00; font-weight: bold; font-size: 1.1rem;'>{trend.upper()}</span> ✅</div>
                            <div style='color: #ffffff; font-weight: 500;'>Swing High: <span style='color: #00ff00;'>₹{swing_high:.2f}</span> <span style='color: #888888;'>on {swing_high_date.strftime('%Y-%m-%d')}</span></div>
                            <div style='color: #ffffff; font-weight: 500;'>Swing Low: <span style='color: #ff0000;'>₹{swing_low:.2f}</span> <span style='color: #888888;'>on {swing_low_date.strftime('%Y-%m-%d')}</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("""
                    **What does this mean?**
                    - **UPTREND**: The stock is generally moving up 📈
                    - **DOWNTREND**: The stock is generally moving down 📉
                    - **Swing High/Low**: Recent highest and lowest prices
                    """)

                with col_anomaly:
                    st.markdown(f"""
                    <div style='background-color: #1e2130; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
                        <h3 style='color: #ffffff; margin: 0 0 1rem 0; font-size: 1.2rem;'>Anomaly Summary</h3>
                        <div style='display: flex; flex-direction: column; gap: 0.5rem;'>
                            <div style='color: #00ff00; font-weight: 500;'>Bullish Anomalies: <span style='color: #ffffff; font-weight: bold;'>{len(df[df['Anomaly_Type'] == 'Bullish'])}</span></div>
                            <div style='color: #ff0000; font-weight: 500;'>Bearish Anomalies: <span style='color: #ffffff; font-weight: bold;'>{len(df[df['Anomaly_Type'] == 'Bearish'])}</span></div>
                            <div style='color: #ffffff; font-weight: 500; margin-top: 0.5rem;'>Average Anomaly Strength: <span style='color: #ffd700; font-weight: bold;'>{df['Anomaly_Strength'].mean():.1f}%</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("""
                    **What is an anomaly?**
                    - A sudden, unusual price move (up or down)
                    - **Bullish**: Big upward jump 🚀
                    - **Bearish**: Big downward drop ⚠️
                    """)
                
                # Display Fibonacci levels
                st.markdown("""
                <div style='background-color: #1e2130; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                    <h3 style='color: #ffffff; margin: 0 0 1rem 0;'>Fibonacci Retracement Levels</h3>
                </div>
                """, unsafe_allow_html=True)

                fib_df = pd.DataFrame.from_dict(fib_levels, orient='index', columns=['Price'])
                st.dataframe(
                    fib_df.style.format({'Price': '₹{:.2f}'}),
                    use_container_width=True
                )
                
                # Display anomalies with more details
                st.markdown("""
                <div style='background-color: #1e2130; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                    <h3 style='color: #ffffff; margin: 0 0 1rem 0;'>Significant Price Movements</h3>
                </div>
                """, unsafe_allow_html=True)

                recent_anomalies = df[df['Is_Anomaly']].tail(5)
                if not recent_anomalies.empty:
                    anomaly_data = recent_anomalies[['Close', 'Z_Score', 'Price_Change_Pct', 'Volume_Ratio']]
                    st.dataframe(
                        anomaly_data.style.format({
                            'Close': '₹{:.2f}',
                            'Z_Score': '{:.2f}',
                            'Price_Change_Pct': '{:+.2f}%',
                            'Volume_Ratio': '{:.2f}x'
                        }).background_gradient(subset=['Price_Change_Pct'], cmap='RdYlGn'),
                        use_container_width=True
                    )
                else:
                    st.info("No significant price movements detected")
                
                # --- Section: Glossary ---
                with st.expander("ℹ️ What do these terms mean?"):
                    st.markdown("""
                    - **Trend**: The general direction the stock price is moving.
                    - **Swing High/Low**: Recent highest/lowest prices.
                    - **Anomaly**: A sudden, unusual price movement.
                    - **RSI**: Measures if a stock is overbought or oversold.
                    - **MACD**: Shows momentum and possible trend changes.
                    - **Fibonacci Levels**: Price levels where the stock might reverse or pause.
                    - **Signal**: Our model's suggestion to buy, sell, or wait.
                    """)

                # --- Section: Next Steps ---
                st.markdown("""
                ---
                #### 💡 What should I do next?
                - Use these insights as a guide, not a guarantee.
                - Consider watching the stock for a few days if unsure.
                - Always do your own research or consult a financial advisor before making big decisions.
                - Investing is risky—never invest money you can't afford to lose!
                """)
                
        except Exception as e:
            st.error(f'Error: {str(e)}')

if __name__ == '__main__':
    main() 