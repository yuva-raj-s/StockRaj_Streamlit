# Indian Stock Market Chatbot

An AI-powered chatbot for Indian stock market information, built with Streamlit and Python.

## Features

- Real-time stock price information
- Market activity monitoring
- Sentiment analysis for stocks
- Market term explanations
- Portfolio and watchlist management
- Interactive chat interface

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run AI_Chat/app.py
```

## Usage

1. **Stock Price Queries**
   - "What's the price of TCS?"
   - "Show me the current value of Reliance"
   - "What's HDFC Bank trading at?"

2. **Market Activity**
   - "Show market activity"
   - "What's the market status?"
   - "Show me Nifty and Sensex"

3. **Sentiment Analysis**
   - "What's the sentiment for Reliance?"
   - "Show me sentiment analysis for TCS"
   - "How's the market sentiment for HDFC Bank?"

4. **Market Terms**
   - "What is IPO?"
   - "Explain FII"
   - "What are circuit limits?"

5. **Portfolio and Watchlist**
   - "Show my portfolio"
   - "What's in my watchlist?"
   - "Show portfolio performance"

## Directory Structure

```
.
├── AI_Chat/
│   └── app.py
├── Portfolio/
│   └── portfolio_data.json
├── Watchlist/
│   └── watchlist.json
├── requirements.txt
└── README.md
```

## Dependencies

- streamlit: Web application framework
- yfinance: Yahoo Finance API wrapper
- pandas: Data manipulation
- numpy: Numerical computing
- textblob: Text processing
- transformers: NLP models
- torch: Deep learning framework
- fuzzywuzzy: String matching
- ta: Technical analysis
- scikit-learn: Machine learning
- tensorflow: Deep learning

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 