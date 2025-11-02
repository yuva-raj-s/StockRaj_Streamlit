import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Add project root to path to allow imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from lstm_model.lstm_prediction import (
    prepare_data,
    create_lstm_model,
    calculate_technical_indicators,
)

# --- Configuration ---
API_BASE_URL = "http://localhost:8000" # This will be your Streamlit app's URL in production
STOCKS_TO_ARCHIVE = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
DATA_ARCHIVE_PATH = os.path.join(project_root, "data_archive")
MODEL_SAVE_PATH = os.path.join(project_root, "models")


def pre_cache_market_data():
    """
    Hits key API endpoints to warm up the cache before market open.
    """
    logging.info("Starting pre-caching of market data...")
    endpoints_to_cache = [
        "/api/dashboard/marquee",
        "/api/dashboard/market-overview",
        "/api/dashboard/top-stocks",
        "/api/market-activity/sector-performance",
        "/api/market-activity/broad-indices",
    ]

    for endpoint in endpoints_to_cache:
        try:
            url = f"{API_BASE_URL}{endpoint}"
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                logging.info(f"Successfully cached endpoint: {endpoint}")
            else:
                logging.warning(f"Failed to cache {endpoint}. Status: {response.status_code}")
        except requests.RequestException as e:
            logging.error(f"Error caching {endpoint}: {e}")

    logging.info("Pre-caching complete.")


def archive_data_and_retrain_model():
    """
    Fetches the latest daily data for key stocks, appends it to an archive,
    and retrains the LSTM model.
    """
    logging.info("Starting data archiving and model retraining...")
    os.makedirs(DATA_ARCHIVE_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    for stock_symbol in STOCKS_TO_ARCHIVE:
        try:
            logging.info(f"Processing {stock_symbol}...")
            archive_file = os.path.join(DATA_ARCHIVE_PATH, f"{stock_symbol}.csv")

            # Fetch latest data
            stock = yf.Ticker(stock_symbol)
            new_data = stock.history(period="5d", interval="1d") # Fetch last 5 days to be safe

            # Load existing archive or create new
            if os.path.exists(archive_file):
                archive_df = pd.read_csv(archive_file, index_col="Date", parse_dates=True)
                # Combine and remove duplicates, keeping the latest entry
                combined_df = pd.concat([archive_df, new_data])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            else:
                combined_df = new_data

            combined_df.sort_index(inplace=True)
            combined_df.to_csv(archive_file)
            logging.info(f"Data for {stock_symbol} archived. Total rows: {len(combined_df)}")

            # Retrain model with the updated data
            logging.info(f"Retraining model for {stock_symbol}...")
            df = calculate_technical_indicators(combined_df.copy())
            X, y, scaler = prepare_data(df, sequence_length=60)

            if X.shape[0] > 0:
                model = create_lstm_model(sequence_length=60)
                model.fit(X, y, epochs=50, batch_size=32, verbose=0)
                
                model_path = os.path.join(MODEL_SAVE_PATH, f"lstm_model_{stock_symbol}.h5")
                model.save(model_path)
                logging.info(f"Model for {stock_symbol} retrained and saved to {model_path}")

        except Exception as e:
            logging.error(f"Failed to process {stock_symbol}: {e}")

    logging.info("Data archiving and model retraining complete.")


def generate_daily_report():
    """
    Generates a summary of the day's market activity and saves it to a file.
    """
    logging.info("Generating EOD market report...")
    
    # This is a placeholder. In a real scenario, you'd fetch data
    # from your API or directly and format it into a report (e.g., Markdown, JSON).
    report_path = os.path.join(project_root, "reports")
    os.makedirs(report_path, exist_ok=True)
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    report_file = os.path.join(report_path, f"market_report_{today_str}.md")

    report_content = f"""
# Market Report for {today_str}

This is an auto-generated report summarizing the day's market activity.

*This feature is a placeholder. Logic to fetch and summarize data would be added here.*

**Key Metrics:**
- NIFTY 50: [Data to be fetched]
- SENSEX: [Data to be fetched]
- Top Gainer: [Data to be fetched]
- Top Loser: [Data to be fetched]
"""
    with open(report_file, "w") as f:
        f.write(report_content)

    logging.info(f"Daily report saved to {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scheduled tasks for StockRaj.")
    parser.add_argument("task", choices=["precache", "archive", "report"], help="The task to run.")
    args = parser.parse_args()

    if args.task == "precache":
        pre_cache_market_data()
    elif args.task == "archive":
        archive_data_and_retrain_model()
    elif args.task == "report":
        generate_daily_report()