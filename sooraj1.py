import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from datetime import datetime, timedelta
from scipy import stats

def fetch_stock_data(ticker, period='2y', interval='1d'):
    """Fetch historical stock data from Yahoo Finance"""
    stock = yf.Ticker(f"{ticker}.NS")
    df = stock.history(period=period, interval=interval)
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].map(mdates.date2num)
    return df

def calculate_indicators(df, timeframe):
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
    
    return df

def calculate_fibonacci_levels(df):
    """Calculate Fibonacci retracement levels based on recent swing highs and lows"""
    # Find the most recent significant swing high and low
    lookback_period = min(60, len(df))  # Use up to 60 periods or available data
    
    # Find swing high (highest high in lookback period)
    swing_high_idx = df['High'].rolling(window=lookback_period).apply(lambda x: x.argmax(), raw=True).iloc[-1]
    swing_high = df.iloc[int(swing_high_idx)]['High']
    swing_high_date = df.iloc[int(swing_high_idx)]['Date']
    
    # Find swing low (lowest low in lookback period)
    swing_low_idx = df['Low'].rolling(window=lookback_period).apply(lambda x: x.argmin(), raw=True).iloc[-1]
    swing_low = df.iloc[int(swing_low_idx)]['Low']
    swing_low_date = df.iloc[int(swing_low_idx)]['Date']
    
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

def generate_signals(df, timeframe):
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
    
    df['Signal'] = ""
    
    # Volume filter (higher volume than average)
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
    
    df.loc[buy_conditions, 'Signal'] = "BUY"
    df.loc[sell_conditions, 'Signal'] = "SELL"
    
    return df

def calculate_regression_trend(df, future_periods=5):
    """Calculate linear regression trend line with future projection"""
    x = np.arange(len(df))
    y = df['Close'].values
    
    # Calculate linear regression
    slope, intercept, _, _, _ = stats.linregress(x, y)
    
    # Create trend line
    df['Trend_Line'] = intercept + slope * x
    
    # Future projection
    future_x = np.arange(len(df), len(df) + future_periods)
    future_y = intercept + slope * future_x
    
    # Combine with original data
    last_date = mdates.num2date(df['Date'].iloc[-1])
    
    if '1d' in df['Interval'].iloc[0]:
        date_step = timedelta(days=1)
    elif '1wk' in df['Interval'].iloc[0]:
        date_step = timedelta(weeks=1)
    else:  # Monthly
        date_step = timedelta(days=30)
    
    future_dates = [last_date + i*date_step for i in range(1, future_periods+1)]
    
    future_df = pd.DataFrame({
        'Date': [mdates.date2num(d) for d in future_dates],
        'Trend_Line': future_y,
        'is_future': True
    })
    
    df['is_future'] = False
    extended_df = pd.concat([df, future_df], ignore_index=True)
    
    return extended_df, slope

def plot_chart(df, ticker, timeframe, trend_slope, fib_levels, trend, swing_high, swing_low, swing_high_date, swing_low_date):
    """Plot candlestick chart with signals, trend line, and Fibonacci levels"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Split into actual and future data
    actual_df = df[~df['is_future']]
    future_df = df[df['is_future']]
    
    # Candlestick plot (actual data only)
    candlestick_ohlc(ax1, actual_df[['Date', 'Open', 'High', 'Low', 'Close']].values,
                    width=0.6, colorup='g', colordown='r', alpha=0.8)
    
    # Plot indicators
    ax1.plot(actual_df['Date'], actual_df['SMA'], 'b-', label=f'SMA {actual_df["SMA_Period"].iloc[0]}', alpha=0.7)
    ax1.plot(actual_df['Date'], actual_df['EMA'], 'm-', label=f'EMA {actual_df["EMA_Period"].iloc[0]}', alpha=0.7)
    ax1.plot(actual_df['Date'], actual_df['VWAP'], 'c-', label='VWAP', alpha=0.7)
    
    # Plot signals
    for idx, row in actual_df[actual_df['Signal'] != ""].iterrows():
        y_pos = row['Low'] * 0.98 if row['Signal'] == "BUY" else row['High'] * 1.02
        color = 'green' if row['Signal'] == "BUY" else 'red'
        ax1.text(row['Date'], y_pos, row['Signal'], 
                fontsize=12, weight='bold', color=color,
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=color))
    
    # Plot trend line (actual + future)
    ax1.plot(df['Date'], df['Trend_Line'], 'k--', linewidth=2, label='Regression Trend')
    
    # Highlight future projection
    ax1.plot(future_df['Date'], future_df['Trend_Line'], 'k:', linewidth=2, label='Future Projection')
    ax1.fill_between(future_df['Date'], 
                    future_df['Trend_Line'] * 0.95, 
                    future_df['Trend_Line'] * 1.05,
                    color='yellow', alpha=0.1)
    
    # Plot Fibonacci levels
    for level, price in fib_levels.items():
        if level in ['0.0', '1.0']:
            # Main levels (0% and 100%)
            ax1.axhline(y=price, color='purple', linestyle='-', alpha=0.7, linewidth=1.5)
            ax1.text(df['Date'].max(), price, f'Fib {level}%', 
                    color='purple', fontsize=10, va='center', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8))
        elif level in ['0.236', '0.382', '0.5', '0.618', '0.786']:
            # Common retracement levels
            ax1.axhline(y=price, color='blue', linestyle='--', alpha=0.5)
            ax1.text(df['Date'].max(), price, f'Fib {level}%', 
                    color='blue', fontsize=8, va='center', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8))
        else:
            # Extension levels
            ax1.axhline(y=price, color='green', linestyle=':', alpha=0.3)
            ax1.text(df['Date'].max(), price, f'Fib {level}%', 
                    color='green', fontsize=8, va='center', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8))
    
    # Mark swing points
    ax1.plot([swing_high_date], [swing_high], 'ro', markersize=8, label='Swing High')
    ax1.plot([swing_low_date], [swing_low], 'go', markersize=8, label='Swing Low')
    
    # Add timeframe and slope information
    tf_map = {'1d': 'Daily', '1wk': 'Weekly', '1mo': 'Monthly'}
    ax1.text(0.02, 0.95, f"{tf_map[timeframe]} | Trend Slope: {trend_slope:.4f} | {trend.capitalize()}",
            transform=ax1.transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Formatting for price chart
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title(f'{ticker} - {tf_map[timeframe]} Swing Trading Signals with Fibonacci Levels', fontsize=16)
    ax1.set_ylabel('Price (₹)')
    ax1.legend(loc='upper left')
    
    # Plot MACD in the lower subplot
    ax2.plot(actual_df['Date'], actual_df['MACD'], 'b-', label='MACD')
    ax2.plot(actual_df['Date'], actual_df['MACD_Signal'], 'r-', label='Signal')
    
    # Color the area between MACD and Signal line
    ax2.fill_between(actual_df['Date'], 
                    actual_df['MACD'], 
                    actual_df['MACD_Signal'],
                    where=(actual_df['MACD'] > actual_df['MACD_Signal']),
                    facecolor='green', alpha=0.3)
    ax2.fill_between(actual_df['Date'], 
                    actual_df['MACD'], 
                    actual_df['MACD_Signal'],
                    where=(actual_df['MACD'] < actual_df['MACD_Signal']),
                    facecolor='red', alpha=0.3)
    
    # Formatting for MACD chart
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.xticks(rotation=45)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_ylabel('MACD')
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

def main():
    ticker = input("Enter Indian stock symbol (e.g., RELIANCE, TCS, HDFCBANK): ") or "RELIANCE"
    timeframe = input("Enter timeframe (1d, 1wk, 1mo): ") or "1d"
    
    # Fetch and process data
    df = fetch_stock_data(ticker, interval=timeframe)
    df['Interval'] = timeframe  # Store timeframe for future projection
    df = calculate_indicators(df, timeframe)
    df = generate_signals(df, timeframe)
    
    # Add indicator periods for display
    if timeframe == '1d':
        df['SMA_Period'] = 20
        df['EMA_Period'] = 50
    elif timeframe == '1wk':
        df['SMA_Period'] = 13
        df['EMA_Period'] = 26
    else:  # Monthly
        df['SMA_Period'] = 6
        df['EMA_Period'] = 12
    
    # Calculate Fibonacci levels
    fib_levels, trend, swing_high, swing_low, swing_high_date, swing_low_date = calculate_fibonacci_levels(df)
    
    # Calculate regression trend
    extended_df, slope = calculate_regression_trend(df)
    
    # Plot results with Fibonacci levels
    plot_chart(extended_df, ticker, timeframe, slope, fib_levels, trend, swing_high, swing_low, swing_high_date, swing_low_date)

    # Print Fibonacci levels for reference
    print("\nFibonacci Retracement Levels:")
    for level, price in sorted(fib_levels.items(), key=lambda x: float(x[0])):
        print(f"{level}%: ₹{price:.2f}")
    
    # Print trend information
    print(f"\nCurrent Trend: {trend.capitalize()}")
    print(f"Swing High: ₹{swing_high:.2f} on {mdates.num2date(swing_high_date).strftime('%Y-%m-%d')}")
    print(f"Swing Low: ₹{swing_low:.2f} on {mdates.num2date(swing_low_date).strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()