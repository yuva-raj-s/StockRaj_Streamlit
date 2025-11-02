from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from datetime import datetime, timedelta
import io
import base64
from typing import Optional

app = FastAPI()

# CORS configuration to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockRequest(BaseModel):
    ticker: str
    period: Optional[str] = "2y"

def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance for Indian stocks"""
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        df = stock.history(period=period, interval="1d")
        if df.empty:
            raise ValueError("No data found for this ticker")
        df.reset_index(inplace=True)
        df['Date'] = df['Date'].map(mdates.date2num)
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for daily timeframe"""
    # Moving Averages
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA_50'] = EMAIndicator(df['Close'], window=50).ema_indicator()
    
    # MACD
    macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # RSI
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    
    # Volatility Stop (ATR based)
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
    df['ATR'] = atr.average_true_range()
    df['VStop_Long'] = df['High'].rolling(14).max() - 3 * df['ATR']
    df['VStop_Short'] = df['Low'].rolling(14).min() + 3 * df['ATR']
    
    # Volume Weighted Average Price
    df['VWAP'] = VolumeWeightedAveragePrice(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        volume=df['Volume'],
        window=20
    ).volume_weighted_average_price()
    
    return df

def calculate_fibonacci_levels(df: pd.DataFrame):
    """Calculate Fibonacci retracement levels based on recent swing highs and lows"""
    lookback_period = min(60, len(df))
    
    swing_high_idx = df['High'].rolling(window=lookback_period).apply(lambda x: x.argmax(), raw=True).iloc[-1]
    swing_high = df.iloc[int(swing_high_idx)]['High']
    swing_high_date = df.iloc[int(swing_high_idx)]['Date']
    
    swing_low_idx = df['Low'].rolling(window=lookback_period).apply(lambda x: x.argmin(), raw=True).iloc[-1]
    swing_low = df.iloc[int(swing_low_idx)]['Low']
    swing_low_date = df.iloc[int(swing_low_idx)]['Date']
    
    if swing_high_date > swing_low_date:
        trend = 'uptrend'
        fib_range = swing_high - swing_low
        base_price = swing_low
    else:
        trend = 'downtrend'
        fib_range = swing_high - swing_low
        base_price = swing_high
    
    fib_levels = {
        '0.0': swing_high if trend == 'downtrend' else swing_low,
        '0.236': base_price + (0.236 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '0.382': base_price + (0.382 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '0.5': base_price + (0.5 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '0.618': base_price + (0.618 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '0.786': base_price + (0.786 * fib_range * (-1 if trend == 'downtrend' else 1)),
        '1.0': swing_low if trend == 'downtrend' else swing_high,
    }
    
    return fib_levels, trend, swing_high, swing_low, swing_high_date, swing_low_date

def calculate_future_projection(df: pd.DataFrame, trend: str, swing_high: float, swing_low: float):
    """Calculate future price projection line"""
    fib_range = swing_high - swing_low
    current_price = df['Close'].iloc[-1]
    
    if trend == 'uptrend':
        future_price = swing_high + 1.618 * fib_range
    else:
        future_price = swing_low - 1.618 * fib_range
    
    future_periods = 30  # Project 30 days into future
    price_diff = future_price - current_price
    slope = price_diff / future_periods
    
    last_date = mdates.num2date(df['Date'].iloc[-1])
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_periods+1)]
    future_prices = [current_price + (i * slope) for i in range(1, future_periods+1)]
    
    future_df = pd.DataFrame({
        'Date': [mdates.date2num(d) for d in future_dates],
        'Projection_Line': future_prices,
        'is_future': True
    })
    
    df['is_future'] = False
    extended_df = pd.concat([df, future_df], ignore_index=True)
    
    return extended_df, slope

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate buy/sell signals"""
    df['Signal'] = ""
    vol_avg = df['Volume'].rolling(20).mean()
    volume_ok = df['Volume'] > (vol_avg * 1.2)
    
    buy_conditions = (
        (df['Close'] > df['EMA_50']) & 
        (df['Close'] > df['VWAP']) & 
        (df['RSI'] > 50) & 
        (df['MACD'] > df['MACD_Signal']) &
        (df['Close'] > df['VStop_Long']) &
        volume_ok
    )
    
    sell_conditions = (
        (df['Close'] < df['EMA_50']) & 
        (df['Close'] < df['VWAP']) & 
        (df['RSI'] < 50) & 
        (df['MACD'] < df['MACD_Signal']) &
        (df['Close'] < df['VStop_Short']) &
        volume_ok
    )
    
    df.loc[buy_conditions, 'Signal'] = "BUY"
    df.loc[sell_conditions, 'Signal'] = "SELL"
    
    return df

def generate_chart_image(df: pd.DataFrame, ticker: str, trend: str, slope: float) -> str:
    """Generate chart image and return as base64 string"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    actual_df = df[~df['is_future']]
    future_df = df[df['is_future']]
    
    # Candlestick plot
    candlestick_ohlc(ax, actual_df[['Date', 'Open', 'High', 'Low', 'Close']].values,
                    width=0.6, colorup='g', colordown='r', alpha=0.8)
    
    # Plot indicators
    ax.plot(actual_df['Date'], actual_df['SMA_20'], 'b-', label='SMA 20', alpha=0.7)
    ax.plot(actual_df['Date'], actual_df['EMA_50'], 'm-', label='EMA 50', alpha=0.7)
    
    # Plot signals
    for idx, row in actual_df[actual_df['Signal'] != ""].iterrows():
        y_pos = row['Low'] * 0.98 if row['Signal'] == "BUY" else row['High'] * 1.02
        color = 'green' if row['Signal'] == "BUY" else 'red'
        ax.text(row['Date'], y_pos, row['Signal'], 
               fontsize=10, weight='bold', color=color,
               ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor=color))
    
    # Future projection line
    ax.plot(future_df['Date'], future_df['Projection_Line'], 'g--', linewidth=2, alpha=0.8, label='Future Projection')
    
    # Chart formatting
    ax.text(0.02, 0.95, f"Trend: {trend.capitalize()} | Projection Slope: {slope:.4f}",
           transform=ax.transAxes, fontsize=10,
           bbox=dict(facecolor='white', alpha=0.8))
    
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f'{ticker} - Daily Trading Signals', fontsize=14)
    ax.set_ylabel('Price (₹)')
    ax.legend(loc='upper left')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    # Convert to base64 string
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

@app.post("/analyze-stock")
async def analyze_stock(request: StockRequest):
    try:
        # Fetch and process data
        df = fetch_stock_data(request.ticker, request.period)
        df = calculate_indicators(df)
        fib_levels, trend, swing_high, swing_low, swing_high_date, swing_low_date = calculate_fibonacci_levels(df)
        extended_df, slope = calculate_future_projection(df, trend, swing_high, swing_low)
        df_with_signals = generate_signals(extended_df)
        
        # Convert dates back to string format for JSON serialization
        df_with_signals['Date'] = df_with_signals['Date'].apply(lambda x: mdates.num2date(x).strftime('%Y-%m-%d'))
        
        # Prepare the data for frontend
        chart_data = {
            "candlestick": df_with_signals[~df_with_signals['is_future']][['Date', 'Open', 'High', 'Low', 'Close']].to_dict('records'),
            "indicators": {
                "sma_20": df_with_signals[~df_with_signals['is_future']][['Date', 'SMA_20']].to_dict('records'),
                "ema_50": df_with_signals[~df_with_signals['is_future']][['Date', 'EMA_50']].to_dict('records'),
                "future_projection": df_with_signals[df_with_signals['is_future']][['Date', 'Projection_Line']].to_dict('records')
            },
            "signals": df_with_signals[~df_with_signals['is_future'] & (df_with_signals['Signal'] != "")][['Date', 'Signal', 'Close']].to_dict('records'),
            "fibonacci_levels": fib_levels,
            "analysis": {
                "trend": trend,
                "current_signal": df_with_signals[~df_with_signals['is_future'] & (df_with_signals['Signal'] != "")].tail(1)['Signal'].values[0] if not df_with_signals[~df_with_signals['is_future'] & (df_with_signals['Signal'] != "")].empty else "NEUTRAL",
                "current_price": float(df['Close'].iloc[-1]),
                "swing_high": float(swing_high),
                "swing_low": float(swing_low),
                "projection_slope": float(slope)
            }
        }
        
        return {
            "status": "success",
            "ticker": request.ticker,
            "data": chart_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)