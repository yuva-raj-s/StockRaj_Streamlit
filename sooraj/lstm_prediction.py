import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from ta.trend import MACD
from ta.momentum import RSIIndicator
import ta
from scipy import stats

def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        df = stock.history(period=period, interval="1d")
        if df.empty:
            raise ValueError("No data found for this ticker")
        return df
    except Exception as e:
        raise ValueError(f"Error fetching data: {str(e)}")

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    # MACD
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # RSI
    rsi = RSIIndicator(df['Close'])
    df['RSI'] = rsi.rsi()
    
    # Bollinger Bands
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'])
    df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'])
    
    # Moving Averages
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    
    return df

def calculate_fibonacci_levels(df: pd.DataFrame):
    """Calculate Fibonacci retracement levels"""
    lookback_period = min(60, len(df))
    
    swing_high = df['High'].rolling(window=lookback_period).max().iloc[-1]
    swing_low = df['Low'].rolling(window=lookback_period).min().iloc[-1]
    
    diff = swing_high - swing_low
    levels = {
        '0.0': swing_low,
        '0.236': swing_low + 0.236 * diff,
        '0.382': swing_low + 0.382 * diff,
        '0.5': swing_low + 0.5 * diff,
        '0.618': swing_low + 0.618 * diff,
        '0.786': swing_low + 0.786 * diff,
        '1.0': swing_high
    }
    
    return levels

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

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate buy/sell signals based on technical indicators with more selective criteria"""
    df['Signal'] = 0  # 0: no signal, 1: buy, -1: sell
    
    # MACD Signal (only on crossover)
    df['MACD_Cross'] = ((df['MACD'] > df['MACD_Signal']) & 
                        (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)))
    df['MACD_Cross_Down'] = ((df['MACD'] < df['MACD_Signal']) & 
                            (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)))
    
    # RSI Signal (only on extreme values with confirmation)
    rsi_oversold = (df['RSI'] < 30) & (df['RSI'].shift(1) >= 30)
    rsi_overbought = (df['RSI'] > 70) & (df['RSI'].shift(1) <= 70)
    
    # Bollinger Bands Signal (with confirmation)
    bb_buy = (df['Close'] < df['BB_Lower']) & (df['Close'].shift(1) >= df['BB_Lower'].shift(1))
    bb_sell = (df['Close'] > df['BB_Upper']) & (df['Close'].shift(1) <= df['BB_Upper'].shift(1))
    
    # Moving Average Crossover
    ma_cross_up = (df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))
    ma_cross_down = (df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))
    
    # Combine signals with confirmation
    df.loc[df['MACD_Cross'] & (df['RSI'] < 60), 'Signal'] = 1  # Buy signal
    df.loc[df['MACD_Cross_Down'] & (df['RSI'] > 40), 'Signal'] = -1  # Sell signal
    
    # Add RSI signals only if they align with trend
    df.loc[rsi_oversold & (df['Close'] > df['SMA_50']), 'Signal'] = 1
    df.loc[rsi_overbought & (df['Close'] < df['SMA_50']), 'Signal'] = -1
    
    # Add Bollinger Bands signals with confirmation
    df.loc[bb_buy & (df['RSI'] < 40), 'Signal'] = 1
    df.loc[bb_sell & (df['RSI'] > 60), 'Signal'] = -1
    
    # Add Moving Average crossover signals
    df.loc[ma_cross_up & (df['Volume'] > df['Volume'].rolling(20).mean()), 'Signal'] = 1
    df.loc[ma_cross_down & (df['Volume'] > df['Volume'].rolling(20).mean()), 'Signal'] = -1
    
    return df

def prepare_data(df: pd.DataFrame, sequence_length: int = 60):
    """Prepare data for LSTM model"""
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def create_lstm_model(sequence_length: int):
    """Create and compile LSTM model"""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_prices(model, last_sequence, scaler, days_to_predict: int = 20):
    """Predict future prices"""
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_to_predict):
        next_pred = model.predict(current_sequence.reshape(1, -1, 1))
        future_predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred[0, 0]
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
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
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_20'],
            name='SMA 20',
            line=dict(color='#ffa500', width=1.5),
            showlegend=True
        ), secondary_y=False)

    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_50'],
            name='SMA 50',
            line=dict(color='#ff00ff', width=1.5),
            showlegend=True
        ), secondary_y=False)

    if 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA_20'],
            name='EMA 20',
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
    
    st.title('Advanced Stock Price Prediction and Analysis')
    
    # Search and Controls Section - 2 columns like tickertape/moneycontrol
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input('Enter Stock Ticker (e.g., RELIANCE):', 'RELIANCE',
                             help="Enter NSE stock symbol (e.g., RELIANCE, TCS, INFY)")
    with col2:
        prediction_days = st.selectbox('Days to Predict', [5, 10, 15, 20, 30], index=3)
    
    # Additional parameters in a collapsible section
    with st.expander("Advanced Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            anomaly_threshold = st.slider('Anomaly Detection Threshold', 2.0, 4.0, 2.5, 0.1)
        with col2:
            sequence_length = st.selectbox('Sequence Length', [30, 60, 90, 120], index=1)
    
    if st.button('Analyze', type='primary'):
        try:
            with st.spinner('Fetching and analyzing data...'):
                # Fetch and process data
                df = fetch_stock_data(ticker)
                df = calculate_technical_indicators(df)
                df = detect_anomalies(df, threshold=anomaly_threshold)
                df = generate_signals(df)
                fib_levels = calculate_fibonacci_levels(df)
                
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
                model = create_lstm_model(sequence_length)
                
                # Train model with progress bar
                progress_bar = st.progress(0)
                epochs = 50
                for epoch in range(epochs):
                    model.fit(X, y, epochs=1, batch_size=32, verbose=0)
                    progress_bar.progress((epoch + 1) / epochs)
                
                # Make predictions
                last_sequence = X[-1]
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
                    st.metric('Predicted Price', f'₹{predictions[-1][0]:.2f}')
                with col3:
                    price_change = ((predictions[-1][0] - df["Close"].iloc[-1]) / df["Close"].iloc[-1]) * 100
                    st.metric('Predicted Change', f'{price_change:+.2f}%', 
                            delta=f'{price_change:+.2f}%',
                            delta_color="normal")
                with col4:
                    st.metric('RSI', f'{df["RSI"].iloc[-1]:.2f}')
                
                # Next 5 Days Predictions Section
                st.markdown("""
                <div style='background-color: #1e2130; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                    <h3 style='color: #ffffff; margin: 0 0 1rem 0;'>Next 5 Days Price Predictions</h3>
                </div>
                """, unsafe_allow_html=True)

                last_date = df.index[-1]
                future_dates = [last_date + timedelta(days=x+1) for x in range(5)]
                current_price = df['Close'].iloc[-1]
                
                prediction_data = []
                for i, (date, pred_price) in enumerate(zip(future_dates, predictions[:5])):
                    daily_change = ((pred_price[0] - current_price) / current_price) * 100 if i == 0 else \
                                 ((pred_price[0] - predictions[i-1][0]) / predictions[i-1][0]) * 100
                    prediction_data.append({
                        'Date': date.strftime('%Y-%m-%d'),
                        'Predicted Price': pred_price[0],
                        'Daily Change': daily_change,
                        'Cumulative Change': ((pred_price[0] - current_price) / current_price) * 100
                    })
                
                prediction_df = pd.DataFrame(prediction_data)
                st.dataframe(
                    prediction_df.style.format({
                        'Predicted Price': '₹{:.2f}',
                        'Daily Change': '{:+.2f}%',
                        'Cumulative Change': '{:+.2f}%'
                    }).background_gradient(subset=['Daily Change', 'Cumulative Change'], 
                                         cmap='RdYlGn'),
                    use_container_width=True
                )
                
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
                    
                    # Add anomaly summary
                    bullish_count = len(recent_anomalies[recent_anomalies['Anomaly_Type'] == 'Bullish'])
                    bearish_count = len(recent_anomalies[recent_anomalies['Anomaly_Type'] == 'Bearish'])
                    
                    st.markdown(f"""
                    <div style='background-color: #1e2130; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                        <h4 style='color: #ffffff; margin: 0 0 0.5rem 0;'>Anomaly Summary</h4>
                        <p style='color: #00ff00; margin: 0;'>Bullish Anomalies: {bullish_count}</p>
                        <p style='color: #ff0000; margin: 0;'>Bearish Anomalies: {bearish_count}</p>
                        <p style='color: #ffffff; margin: 0.5rem 0 0 0;'>Average Anomaly Strength: {recent_anomalies['Anomaly_Strength'].mean():.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No significant price movements detected")
                
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
                
        except Exception as e:
            st.error(f'Error: {str(e)}')

if __name__ == '__main__':
    main() 