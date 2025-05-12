import logging
import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from GoogleNews import GoogleNews
from transformers import pipeline
from datetime import datetime, timedelta
import matplotlib
import yfinance as yf
import io
import base64
from PIL import Image
matplotlib.use('Agg')

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize sentiment analysis model
SENTIMENT_ANALYSIS_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
device = 0 if torch.cuda.is_available() else -1  # Use integer index

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_ANALYSIS_MODEL,
    device=device
)
logging.info("Model initialized successfully")

# Indian stock ticker mapping
COMMON_TICKERS = {
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "hdfc": "HDFC.NS",
    "sbi": "SBIN.NS",
    "kotak bank": "KOTAKBANK.NS",
    "axis bank": "AXISBANK.NS",
    "wipro": "WIPRO.NS",
    "tech mahindra": "TECHM.NS",
    "asian paints": "ASIANPAINT.NS",
    "bajaj auto": "BAJAJ-AUTO.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "hindalco": "HINDALCO.NS",
    "itc": "ITC.NS",
    "larsen": "LT.NS",
    "l&t": "LT.NS",
    "larsen and toubro": "LT.NS",
    "maruti": "MARUTI.NS",
    "nestle": "NESTLEIND.NS",
    "ongc": "ONGC.NS",
    "power grid": "POWERGRID.NS",
    "sun pharma": "SUNPHARMA.NS",
    "tata motors": "TATAMOTORS.NS",
    "tata steel": "TATASTEEL.NS",
    "ultracemco": "ULTRACEMCO.NS",
    "upl": "UPL.NS",
    "zeel": "ZEEL.NS",
    "zee": "ZEEL.NS",
    "zee entertainment": "ZEEL.NS"
}

def fetch_articles(query, max_articles=10):
    try:
        logging.info(f"Fetching up to {max_articles} articles for query: '{query}'")
        googlenews = GoogleNews(lang="en")
        googlenews.search(query)
        
        articles = googlenews.result()
        
        page = 2
        while len(articles) < max_articles and page <= 10:
            logging.info(f"Fetched {len(articles)} articles so far. Getting page {page}...")
            googlenews.get_page(page)
            page_results = googlenews.result()
            
            if not page_results:
                logging.info(f"No more results found after page {page-1}")
                break
                
            articles.extend(page_results)
            page += 1
            
        articles = articles[:max_articles]
        
        logging.info(f"Successfully fetched {len(articles)} articles")
        return articles
    except Exception as e:
        logging.error(f"Error while searching articles for query: '{query}'. Error: {e}")
        st.error(f"Unable to search articles for query: '{query}'. Try again later...")
        return []

def analyze_article_sentiment(article):
    logging.info(f"Analyzing sentiment for article: {article['title']}")
    sentiment = sentiment_analyzer(article["desc"])[0]
    article["sentiment"] = sentiment
    return article

def calculate_time_weight(article_date_str):
    """
    기사 시간 기준으로 가중치 계산 
    - 1시간 내 기사는 24% 가중치
    - 시간이 지날수록 1%씩 감소 (최소 1%)
    - 예: 1시간 내 기사 = 24%, 10시간 전 기사 = 15%, 24시간 전 기사 = 1%
    - 24시간 이상이면 1%로 고정
    """
    try:
        # 기사 날짜 문자열 파싱 (다양한 형식 처리)
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %z',  # 기본 GoogleNews 형식
            '%Y-%m-%d %H:%M:%S',
            '%a, %d %b %Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S%z',
            '%a %b %d, %Y',
            '%d %b %Y'
        ]
        
        parsed_date = None
        for format_str in date_formats:
            try:
                parsed_date = datetime.strptime(article_date_str, format_str)
                break
            except ValueError:
                continue
        
        # 어떤 형식으로도 파싱할 수 없으면 현재 시간 기준 24시간 전으로 가정
        if parsed_date is None:
            logging.warning(f"Could not parse date: {article_date_str}, using default 24h ago")
            return 0.01  # 최소 가중치 1%
            
        # 현재 시간과의 차이 계산 (시간 단위)
        now = datetime.now()
        if parsed_date.tzinfo is not None:
            now = now.replace(tzinfo=parsed_date.tzinfo)
            
        hours_diff = (now - parsed_date).total_seconds() / 3600
        
        # 24시간 이내인 경우만 고려
        if hours_diff < 1:  # 1시간 이내
            return 0.24  # 24% 가중치
        elif hours_diff < 24:  # 1~23시간
            # 1시간당 1%씩 감소 (1시간 = 24%, 2시간 = 23%, ...)
            return max(0.01, 0.24 - ((hours_diff - 1) * 0.01))
        else:
            return 0.01  # 24시간 이상 지난 기사는 1% 가중치
    except Exception as e:
        logging.error(f"Error calculating time weight: {e}")
        return 0.01  # 오류 발생 시 최소 가중치 적용

def calculate_sentiment_score(sentiment_label, time_weight):
    """
    감성 레이블에 따른 기본 점수 계산 및 시간 가중치 적용
    - positive: +3점
    - neutral: 0점
    - negative: -3점
    
    시간 가중치는 백분율로 적용 (기본 점수에 가중치 % 만큼 추가)
    예: 
    - 1시간 내 긍정 기사: 3점 + (3 * 24%) = 3 + 0.72 = 3.72점
    - 10시간 전 부정 기사: -3점 + (-3 * 15%) = -3 - 0.45 = -3.45점
    """
    base_score = {
        'positive': 3,
        'neutral': 0,
        'negative': -3
    }.get(sentiment_label, 0)
    
    # 가중치를 적용한 추가 점수 계산
    weighted_addition = base_score * time_weight
    
    return base_score, weighted_addition

def get_stock_ticker(asset_name):
    """
    Get stock ticker symbol from asset name
    """
    logging.info(f"Identifying ticker for: {asset_name}")
    
    # Convert to lowercase for mapping check
    asset_lower = asset_name.lower().strip()
    
    # Check if input is already a ticker symbol
    if asset_name.isupper() and 2 <= len(asset_name) <= 6:
        # Add .NS suffix for Indian stocks if not present
        if not asset_name.endswith('.NS'):
            ticker = f"{asset_name}.NS"
        else:
            ticker = asset_name
        logging.info(f"Input appears to be a ticker symbol: {ticker}")
        return ticker
    
    # Check common tickers mapping
    if asset_lower in COMMON_TICKERS:
        ticker = COMMON_TICKERS[asset_lower]
        logging.info(f"Found ticker in common tickers map: {ticker}")
        return ticker
    
    # Try searching by company name
    try:
        search_results = yf.Ticker(asset_name)
        info = search_results.info
        if info and 'symbol' in info:
            ticker = info['symbol']
            # Add .NS suffix for Indian stocks if not present
            if not ticker.endswith('.NS'):
                ticker = f"{ticker}.NS"
            logging.info(f"Found ticker from search: {ticker}")
            return ticker
    except Exception as e:
        logging.debug(f"Search failed: {e}")
    
    logging.warning(f"Could not identify valid ticker for: {asset_name}")
    return None

def create_stock_chart(ticker, period="1mo"):
    """
    Create stock price chart
    """
    try:
        logging.info(f"Fetching stock data for {ticker}")
        stock = yf.Ticker(ticker)
        
        # Get historical data
        hist = stock.history(period=period)
        
        if len(hist) == 0:
            logging.warning(f"No stock data found for ticker: {ticker}")
            return None
            
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot closing price
        ax.plot(hist.index, hist['Close'], label='Close Price', color='blue')
        
        # Add 20-day moving average if enough data
        if len(hist) > 20:
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            ax.plot(hist.index, hist['MA20'], label='20-day MA', color='orange')
        
        # Add volume subplot if available
        if 'Volume' in hist.columns and not hist['Volume'].isna().all():
            ax2 = ax.twinx()
            ax2.bar(hist.index, hist['Volume'], alpha=0.3, color='gray', label='Volume')
            ax2.set_ylabel('Volume')
            
            # Add legend for both plots
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
            ax.legend(loc='upper left')
        
        # Style the plot
        ax.set_title(f"{ticker} Stock Price")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        chart_path = f"stock_chart_{ticker.replace('-', '_').replace('.', '_')}.png"
        plt.savefig(chart_path)
        plt.close()
        
        logging.info(f"Stock chart created: {chart_path}")
        return chart_path
    except Exception as e:
        logging.error(f"Error creating stock chart for {ticker}: {e}")
        return None

def sentiment_bar(positive, neutral, negative):
    fig, ax = plt.subplots(figsize=(5, 0.5))
    ax.barh([0], positive, color='#28a745', label='Positive')
    ax.barh([0], neutral, left=positive, color='#6c757d', label='Neutral')
    ax.barh([0], negative, left=positive+neutral, color='#dc3545', label='Negative')
    ax.set_xlim(0, 100)
    ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img

def analyze_asset_sentiment(asset_name):
    logging.info(f"Starting sentiment analysis for asset: {asset_name}")
    articles = fetch_articles(asset_name, max_articles=30)
    analyzed_articles = [analyze_article_sentiment(article) for article in articles]
    for article in analyzed_articles:
        time_weight = calculate_time_weight(article["date"])
        article["time_weight"] = time_weight
        sentiment_label = article["sentiment"]["label"]
        base_score, weighted_addition = calculate_sentiment_score(sentiment_label, time_weight)
        article["base_score"] = base_score
        article["weighted_addition"] = weighted_addition
        article["total_score"] = base_score + weighted_addition
    # Sentiment summary
    total_articles = len(analyzed_articles)
    positive_count = sum(1 for a in analyzed_articles if a["sentiment"]["label"] == "positive")
    neutral_count = sum(1 for a in analyzed_articles if a["sentiment"]["label"] == "neutral")
    negative_count = sum(1 for a in analyzed_articles if a["sentiment"]["label"] == "negative")
    positive_pct = (positive_count / total_articles) * 100 if total_articles > 0 else 0
    neutral_pct = (neutral_count / total_articles) * 100 if total_articles > 0 else 0
    negative_pct = (negative_count / total_articles) * 100 if total_articles > 0 else 0
    avg_sentiment_score = sum(a["total_score"] for a in analyzed_articles) / total_articles if total_articles > 0 else 0
    # Horizontal bar image
    sentiment_bar_img = sentiment_bar(positive_pct, neutral_pct, negative_pct)
    # Compose summary text
    summary = f"""
    **News Sentiment:** {'🟢 Positive' if avg_sentiment_score > 1 else '⚪ Neutral' if avg_sentiment_score > -1 else '🔴 Negative'}  
    Positive {positive_pct:.0f}% | Neutral {neutral_pct:.0f}% | Negative {negative_pct:.0f}%
    """
    # Compose gauge (reuse existing gauge image)
    sentiment_summary = create_sentiment_summary(analyzed_articles, asset_name)
    # Stock chart
    stock_chart = None
    ticker = get_stock_ticker(asset_name)
    if ticker:
        stock_chart = create_stock_chart(ticker)
    # Return: table, summary markdown, gauge image, chart, ticker
    return (
        convert_to_dataframe(analyzed_articles),
        summary,
        sentiment_bar_img,
        sentiment_summary,
        stock_chart,
        ticker
    )

def create_sentiment_summary(analyzed_articles, asset_name):
    """
    Create sentiment analysis summary with market sentiment card and sentiment gauge
    """
    total_articles = len(analyzed_articles)
    positive_count = sum(1 for a in analyzed_articles if a["sentiment"]["label"] == "positive")
    neutral_count = sum(1 for a in analyzed_articles if a["sentiment"]["label"] == "neutral")
    negative_count = sum(1 for a in analyzed_articles if a["sentiment"]["label"] == "negative")
    
    # Calculate percentages
    positive_pct = (positive_count / total_articles) * 100 if total_articles > 0 else 0
    neutral_pct = (neutral_count / total_articles) * 100 if total_articles > 0 else 0
    negative_pct = (negative_count / total_articles) * 100 if total_articles > 0 else 0
    
    # Calculate weighted sentiment score
    weighted_score_sum = sum(a["total_score"] for a in analyzed_articles)
    avg_sentiment_score = weighted_score_sum / total_articles if total_articles > 0 else 0
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    
    # 1. Market Sentiment Card (Pie Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [positive_pct, neutral_pct, negative_pct]
    colors = ['green', 'gray', 'red']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    ax1.set_title('Market Sentiment Distribution')
    
    # Add summary text
    summary_text = f"""
    Total Articles: {total_articles}
    Positive: {positive_count} ({positive_pct:.1f}%)
    Neutral: {neutral_count} ({neutral_pct:.1f}%)
    Negative: {negative_count} ({negative_pct:.1f}%)
    """
    plt.figtext(0.25, 0.01, summary_text, ha='center', fontsize=10, 
                bbox={"facecolor":"lightgray", "alpha":0.3, "pad":5})
    
    # 2. Sentiment Gauge
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create gauge chart
    gauge_min = -3  # Most negative
    gauge_max = 3   # Most positive
    gauge_value = avg_sentiment_score
    
    # Normalize value to 0-1 range for gauge
    normalized_value = (gauge_value - gauge_min) / (gauge_max - gauge_min)
    
    # Create gauge
    gauge = plt.Circle((0.5, 0.5), 0.4, transform=ax2.transAxes, fill=False)
    ax2.add_patch(gauge)
    
    # Add gauge needle
    angle = normalized_value * 180  # Convert to degrees
    rad = np.radians(angle)
    x = 0.5 + 0.35 * np.cos(rad)
    y = 0.5 + 0.35 * np.sin(rad)
    ax2.plot([0.5, x], [0.5, y], 'r-', linewidth=2)
    
    # Add gauge labels
    ax2.text(0.5, 0.9, 'Bullish', ha='center', va='center')
    ax2.text(0.1, 0.5, 'Bearish', ha='center', va='center')
    ax2.text(0.5, 0.1, 'Neutral', ha='center', va='center')
    
    # Add current value
    sentiment_label = "Bullish" if gauge_value > 1 else "Bearish" if gauge_value < -1 else "Neutral"
    ax2.text(0.5, 0.5, f'{gauge_value:.2f}\n({sentiment_label})', 
             ha='center', va='center', fontsize=12)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Sentiment Gauge')
    
    plt.tight_layout()
    
    # Save the figure
    fig_path = f"sentiment_summary_{asset_name.replace(' ', '_')}.png"
    plt.savefig(fig_path)
    plt.close()
    
    return fig_path

def convert_to_dataframe(analyzed_articles):
    # Sort articles by date in descending order (newest first) and take top 10
    sorted_articles = sorted(analyzed_articles, key=lambda x: x.get("date", ""), reverse=True)[:10]
    
    df = pd.DataFrame(sorted_articles)
    # Sentiment as plain text with emoji
    def sentiment_badge(sentiment):
        if sentiment == "positive":
            return "🟢 Positive"
        elif sentiment == "neutral":
            return "⚪ Neutral"
        elif sentiment == "negative":
            return "🔴 Negative"
        return sentiment
    df["Sentiment"] = df["sentiment"].apply(lambda x: sentiment_badge(x["label"]))
    # Title as plain text (no HTML)
    df["Title"] = df["title"]
    df["Description"] = df["desc"]
    df["Date"] = df["date"]
    df["Base Score"] = df["base_score"].apply(lambda x: f"{x:+.2f}")
    df["Weight"] = df["time_weight"].apply(lambda x: f"{x*10:.1f}x")
    df["Total Score"] = df["total_score"].apply(lambda x: f"{x:+.2f}")
    return df[["Sentiment", "Title", "Description", "Date", "Base Score", "Weight", "Total Score"]]

def main():
    st.title("Indian Stock Market Sentiment Analysis")
    st.write("Enter a stock symbol or company name to analyze market sentiment from recent news articles!")
    
    # Input section
    asset_name = st.text_input("Stock Symbol/Name", 
                              placeholder="Enter stock symbol (e.g., RELIANCE, TCS, INFY) or company name...")
    
    if asset_name:
        try:
            # Analyze sentiment
            articles_df, sentiment_summary, sentiment_bar_img, sentiment_gauge, stock_chart, ticker = analyze_asset_sentiment(asset_name)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Market Sentiment")
                st.markdown(sentiment_summary)
                st.image(sentiment_bar_img)
            
            with col2:
                st.markdown("### Sentiment Gauge")
                st.image(sentiment_gauge)
            
            if stock_chart:
                st.markdown("### Stock Price Chart")
                st.image(stock_chart)
                if ticker:
                    st.text(f"Ticker Symbol: {ticker}")
            
            st.markdown("### Articles and Sentiment Analysis")
            st.dataframe(articles_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred while analyzing sentiment: {str(e)}")
            logging.error(f"Error in sentiment analysis: {str(e)}")

if __name__ == "__main__":
    main()
