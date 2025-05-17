import logging
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GoogleNews import GoogleNews
from textblob import TextBlob
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
    """Analyze sentiment using TextBlob"""
    logging.info(f"Analyzing sentiment for article: {article['title']}")
    text = article["desc"]
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # Convert polarity to sentiment label
    if polarity > 0.1:
        sentiment = {"label": "positive", "score": polarity}
    elif polarity < -0.1:
        sentiment = {"label": "negative", "score": abs(polarity)}
    else:
        sentiment = {"label": "neutral", "score": 0.5}
    
    article["sentiment"] = sentiment
    return article

def calculate_time_weight(article_date_str):
    """Calculate time-based weight for articles"""
    try:
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %z',
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
        
        if parsed_date is None:
            logging.warning(f"Could not parse date: {article_date_str}, using default 24h ago")
            return 0.01
            
        now = datetime.now()
        if parsed_date.tzinfo is not None:
            now = now.replace(tzinfo=parsed_date.tzinfo)
            
        hours_diff = (now - parsed_date).total_seconds() / 3600
        
        if hours_diff < 1:
            return 0.24
        elif hours_diff < 24:
            return max(0.01, 0.24 - ((hours_diff - 1) * 0.01))
        else:
            return 0.01
    except Exception as e:
        logging.error(f"Error calculating time weight: {e}")
        return 0.01

def calculate_sentiment_score(sentiment_label, time_weight):
    """Calculate sentiment score with time weight"""
    base_score = {
        'positive': 3,
        'neutral': 0,
        'negative': -3
    }.get(sentiment_label, 0)
    
    weighted_addition = base_score * time_weight
    return base_score, weighted_addition

def get_stock_ticker(asset_name):
    """Get stock ticker symbol from asset name"""
    logging.info(f"Identifying ticker for: {asset_name}")
    
    asset_lower = asset_name.lower().strip()
    
    if asset_name.isupper() and 2 <= len(asset_name) <= 6:
        if not asset_name.endswith('.NS'):
            ticker = f"{asset_name}.NS"
        else:
            ticker = asset_name
        logging.info(f"Input appears to be a ticker symbol: {ticker}")
        return ticker
    
    if asset_lower in COMMON_TICKERS:
        ticker = COMMON_TICKERS[asset_lower]
        logging.info(f"Found ticker in common tickers map: {ticker}")
        return ticker
    
    try:
        search_results = yf.Ticker(asset_name)
        info = search_results.info
        if info and 'symbol' in info:
            ticker = info['symbol']
            if not ticker.endswith('.NS'):
                ticker = f"{ticker}.NS"
            logging.info(f"Found ticker from search: {ticker}")
            return ticker
    except Exception as e:
        logging.debug(f"Search failed: {e}")
    
    logging.warning(f"Could not identify valid ticker for: {asset_name}")
    return None

def create_stock_chart(ticker, period="1mo"):
    """Create stock price chart"""
    try:
        logging.info(f"Fetching stock data for {ticker}")
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if len(hist) == 0:
            logging.warning(f"No stock data found for ticker: {ticker}")
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hist.index, hist['Close'], label='Close Price', color='blue')
        
        if len(hist) > 20:
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            ax.plot(hist.index, hist['MA20'], label='20-day MA', color='orange')
        
        if 'Volume' in hist.columns and not hist['Volume'].isna().all():
            ax2 = ax.twinx()
            ax2.bar(hist.index, hist['Volume'], alpha=0.3, color='gray', label='Volume')
            ax2.set_ylabel('Volume')
        
        ax.set_title(f'{ticker} Stock Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf
    except Exception as e:
        logging.error(f"Error creating stock chart: {e}")
        return None

def sentiment_bar(positive, neutral, negative):
    """Create sentiment distribution bar chart"""
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        sentiments = ['Positive', 'Neutral', 'Negative']
        values = [positive, neutral, negative]
        colors = ['green', 'gray', 'red']
        
        ax.bar(sentiments, values, color=colors)
        ax.set_title('Sentiment Distribution')
        ax.set_ylabel('Number of Articles')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf
    except Exception as e:
        logging.error(f"Error creating sentiment bar chart: {e}")
        return None

def analyze_asset_sentiment(asset_name):
    """Analyze sentiment for a given asset"""
    try:
        ticker = get_stock_ticker(asset_name)
        if not ticker:
            st.error(f"Could not find ticker for {asset_name}")
            return None, None, None, None, None, None
        
        # Fetch articles
        articles = fetch_articles(asset_name)
        if not articles:
            st.warning(f"No articles found for {asset_name}")
            return None, None, None, None, None, None
        
        # Analyze sentiment for each article
        analyzed_articles = [analyze_article_sentiment(article) for article in articles]
        
        # Create sentiment summary
        sentiment_summary = create_sentiment_summary(analyzed_articles, asset_name)
        
        # Create sentiment visualization
        positive = sum(1 for a in analyzed_articles if a['sentiment']['label'] == 'positive')
        neutral = sum(1 for a in analyzed_articles if a['sentiment']['label'] == 'neutral')
        negative = sum(1 for a in analyzed_articles if a['sentiment']['label'] == 'negative')
        
        sentiment_bar_img = sentiment_bar(positive, neutral, negative)
        
        # Create stock chart
        stock_chart = create_stock_chart(ticker)
        
        # Convert articles to DataFrame
        articles_df = convert_to_dataframe(analyzed_articles)
        
        return articles_df, sentiment_summary, sentiment_bar_img, None, stock_chart, ticker
        
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}")
        st.error(f"Error analyzing sentiment: {str(e)}")
        return None, None, None, None, None, None

def create_sentiment_summary(analyzed_articles, asset_name):
    """Create a summary of sentiment analysis"""
    try:
        positive = sum(1 for a in analyzed_articles if a['sentiment']['label'] == 'positive')
        neutral = sum(1 for a in analyzed_articles if a['sentiment']['label'] == 'neutral')
        negative = sum(1 for a in analyzed_articles if a['sentiment']['label'] == 'negative')
        
        total = len(analyzed_articles)
        positive_pct = (positive / total) * 100 if total > 0 else 0
        negative_pct = (negative / total) * 100 if total > 0 else 0
        
        summary = f"""
        ### Sentiment Analysis Summary for {asset_name}
        
        - Total Articles Analyzed: {total}
        - Positive Sentiment: {positive} ({positive_pct:.1f}%)
        - Neutral Sentiment: {neutral} ({100 - positive_pct - negative_pct:.1f}%)
        - Negative Sentiment: {negative} ({negative_pct:.1f}%)
        
        Overall sentiment is {'positive' if positive_pct > negative_pct else 'negative' if negative_pct > positive_pct else 'neutral'}.
        """
        
        return summary
    except Exception as e:
        logging.error(f"Error creating sentiment summary: {e}")
        return "Error creating sentiment summary"

def convert_to_dataframe(analyzed_articles):
    """Convert analyzed articles to DataFrame"""
    try:
        df = pd.DataFrame([{
            'Date': article['date'],
            'Title': article['title'],
            'Description': article['desc'],
            'Sentiment': article['sentiment']['label'].capitalize(),
            'Confidence': f"{article['sentiment']['score']*100:.1f}%"
        } for article in analyzed_articles])
        
        return df
    except Exception as e:
        logging.error(f"Error converting to DataFrame: {e}")
        return pd.DataFrame()

def main():
    st.title("Stock Sentiment Analysis")
    asset_name = st.text_input("Enter stock name or ticker:", "RELIANCE")
    
    if st.button("Analyze"):
        articles_df, sentiment_summary, sentiment_bar_img, sentiment_gauge, stock_chart, ticker = analyze_asset_sentiment(asset_name)
        
        if articles_df is not None:
            st.markdown(sentiment_summary)
            
            if sentiment_bar_img:
                st.image(sentiment_bar_img, caption="Sentiment Distribution")
            
            if stock_chart:
                st.image(stock_chart, caption=f"{ticker} Stock Price")
            
            st.dataframe(articles_df, use_container_width=True)

if __name__ == "__main__":
    main()
