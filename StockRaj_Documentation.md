# StockRaj - Indian Stock Market Analysis Platform
## Comprehensive Documentation

### 1. Project Overview
StockRaj is a comprehensive Indian stock market analysis platform built using Streamlit. It provides real-time market data, portfolio management, watchlist tracking, and AI-powered analysis tools. The platform features a modern dark theme UI and is designed to help investors make informed decisions.

### 2. Core Components

#### 2.1 Main Application (app.py)
The main application serves as the entry point and orchestrator for the entire platform. It features:
- Modern dark theme UI with custom CSS styling
- Responsive sidebar navigation
- Session state management
- Integration of all major components

#### 2.2 Dashboard (Dashboard/streamlit_dashboard.py)
The dashboard provides a comprehensive market overview with:
- Live market ticker for major stocks
- Market overview with major indices (NIFTY 50, SENSEX, NIFTY BANK)
- Top Indian stocks performance
- Latest financial news with refresh capability
- Real-time data updates

#### 2.3 Market Activity (Market_Activity/market_activity.py)
This component offers detailed market analysis with:
- Market Overview tab:
  - Market pulse with key metrics
  - Broad indices performance
  - Market volatility indicators
- Sector Analysis tab:
  - Sector-wise performance metrics
  - Sector trends visualization
  - Detailed sector analysis with top companies
  - Market cap and price metrics

#### 2.4 Watchlist (Watchlist/watchlist_page.py)
The watchlist feature allows users to:
- Track favorite stocks
- Monitor price movements
- Set price alerts
- View detailed stock information
- Manage watchlist entries

#### 2.5 Portfolio (Portfolio/portfolio.py)
A comprehensive portfolio management system with:
- Portfolio Overview:
  - Total value and investment tracking
  - P&L calculations
  - Holdings table
  - Performance charts
- Transaction Management:
  - Add buy/sell transactions
  - Transaction history
  - Price and quantity tracking
- Investment Goals:
  - Set financial goals
  - Track progress
  - Priority management

#### 2.6 AI Analysis (AI_Analysis/ai_analysis.py)
Advanced AI-powered analysis tools including:
- Technical Indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - SMA (Simple Moving Averages)
  - Bollinger Bands
- Sentiment Analysis:
  - Market sentiment tracking
  - News analysis
  - Sentiment visualization
- Prediction & Signals:
  - Price predictions
  - Trading signals
  - Market trend analysis

#### 2.6.1 LSTM Price Predictions
The platform implements advanced LSTM (Long Short-Term Memory) neural networks for price predictions:

1. Data Preprocessing:
   - Historical price data collection (OHLCV - Open, High, Low, Close, Volume)
   - Feature engineering including:
     - Technical indicators (RSI, MACD, Bollinger Bands)
     - Price momentum indicators
     - Volume-based indicators
   - Data normalization using MinMaxScaler
   - Sequence creation for time series analysis

2. LSTM Model Architecture:
   - Multiple LSTM layers with dropout for regularization
   - Dense layers for final predictions
   - Custom loss functions for financial time series
   - Hyperparameter optimization for:
     - Sequence length
     - Number of LSTM units
     - Dropout rates
     - Learning rate

3. Training Process:
   - Split data into training (70%), validation (15%), and testing (15%)
   - Batch processing with dynamic batch sizes
   - Early stopping to prevent overfitting
   - Model checkpointing for best weights
   - Cross-validation for robust performance

4. Prediction Features:
   - Short-term price predictions (1-5 days)
   - Trend direction prediction
   - Volatility forecasting
   - Confidence intervals for predictions
   - Multiple timeframe analysis

5. Performance Metrics:
   - Mean Absolute Error (MAE)
   - Root Mean Square Error (RMSE)
   - Directional Accuracy
   - Sharpe Ratio for prediction quality
   - Maximum Drawdown analysis

#### 2.6.2 Model Specifications
1. LSTM Model Details:
   - Architecture: Stacked LSTM with 3 layers
   - Layer Configuration:
     - LSTM Layer 1: 128 units with 0.2 dropout
     - LSTM Layer 2: 64 units with 0.2 dropout
     - LSTM Layer 3: 32 units with 0.2 dropout
     - Dense Layer: 1 unit for price prediction
   - Activation Functions:
     - LSTM layers: tanh
     - Dense layer: linear
   - Optimizer: Adam with learning rate 0.001
   - Loss Function: Mean Squared Error (MSE)
   - Batch Size: 32
   - Epochs: 100 with early stopping

2. Technical Analysis Models:
   - RSI (Relative Strength Index):
     - Period: 14 days
     - Overbought threshold: 70
     - Oversold threshold: 30
   - MACD (Moving Average Convergence Divergence):
     - Fast period: 12
     - Slow period: 26
     - Signal period: 9
   - Bollinger Bands:
     - Period: 20
     - Standard deviation: 2
   - Moving Averages:
     - SMA: 20, 50, 200 days
     - EMA: 12, 26 days

3. Pattern Recognition Models:
   - Candlestick Patterns:
     - Doji
     - Hammer
     - Engulfing
     - Morning/Evening Star
   - Chart Patterns:
     - Head and Shoulders
     - Double Top/Bottom
     - Triangle Patterns
     - Flag and Pennant

#### 2.6.3 Advanced Sentiment Analysis
1. News Sentiment Analysis:
   - Model: FinBERT (Financial BERT)
     - Pre-trained on financial text
     - Fine-tuned for Indian market context
     - Sentiment categories: Positive, Negative, Neutral
   - Features:
     - Article-level sentiment
     - Sentence-level granularity
     - Entity-specific sentiment
     - Temporal sentiment tracking

2. Social Media Sentiment:
   - Twitter Analysis:
     - Real-time tweet collection
     - Hashtag tracking
     - User influence weighting
     - Sentiment aggregation
   - Reddit Analysis:
     - Subreddit monitoring
     - Comment sentiment
     - Post engagement metrics
     - Community sentiment trends

3. Market Sentiment Indicators:
   - Fear & Greed Index:
     - Market volatility
     - Market momentum
     - Social media sentiment
     - News sentiment
     - Put/Call ratio
   - VIX (Volatility Index) Analysis:
     - Historical VIX patterns
     - Correlation with market movements
     - Volatility forecasting

4. Sentiment Aggregation:
   - Weighted Scoring System:
     - News articles: 40%
     - Social media: 30%
     - Technical indicators: 20%
     - Market data: 10%
   - Temporal Weighting:
     - Recent sentiment: higher weight
     - Historical sentiment: lower weight
   - Source Credibility:
     - Financial news outlets: higher weight
     - Social media: lower weight
     - Expert opinions: medium weight

5. Sentiment Visualization:
   - Sentiment Timeline:
     - Historical sentiment trends
     - Event correlation
     - Pattern identification
   - Sentiment Heatmaps:
     - Sector-wise sentiment
     - Stock-specific sentiment
     - Market-wide sentiment
   - Sentiment Gauges:
     - Real-time sentiment indicators
     - Threshold alerts
     - Trend indicators

6. Sentiment-Based Trading Signals:
   - Signal Generation:
     - Sentiment threshold triggers
     - Trend confirmation
     - Volume confirmation
   - Risk Management:
     - Sentiment-based stop-loss
     - Position sizing
     - Entry/exit timing
   - Performance Metrics:
     - Signal accuracy
     - Return on signals
     - Risk-adjusted returns

7. Integration with Other Models:
   - Technical Analysis:
     - Sentiment confirmation of technical signals
     - Divergence detection
     - Trend strength assessment
   - Price Prediction:
     - Sentiment as a feature
     - Model ensemble weighting
     - Prediction confidence adjustment
   - Portfolio Management:
     - Sentiment-based rebalancing
     - Risk adjustment
     - Asset allocation

#### 2.7 AI Chat (AI_Chat/app.py)
The AI Chat component is a sophisticated natural language interface with multiple capabilities:

1. Core Architecture:
   - Transformer-based language models
   - Multi-model ensemble approach
   - Real-time data integration
   - Context-aware responses
   - Memory management for conversation history

2. Natural Language Processing:
   - Intent Classification:
     - Price queries
     - Technical analysis requests
     - Portfolio management
     - Market news requests
     - General market queries
   - Entity Recognition:
     - Stock symbols
     - Company names
     - Market terms
     - Numerical values
   - Sentiment Analysis:
     - News sentiment
     - Market sentiment
     - Social media sentiment

3. Query Processing Pipeline:
   a. Input Processing:
      - Text normalization
      - Spell correction
      - Entity extraction
      - Context analysis
   
   b. Intent Classification:
      - Multi-class classification
      - Confidence scoring
      - Fallback handling
   
   c. Data Retrieval:
      - Real-time market data
      - Historical data
      - News articles
      - Technical indicators
   
   d. Response Generation:
      - Natural language generation
      - Data formatting
      - Visualization integration
      - Error handling

4. Specialized Features:
   a. Portfolio Analysis:
      - Performance tracking
      - Risk assessment
      - Portfolio optimization suggestions
      - Transaction history analysis
   
   b. Watchlist Management:
      - Stock addition/removal
      - Price alerts
      - Performance tracking
      - News monitoring
   
   c. Market Analysis:
      - Sector performance
      - Market trends
      - Volatility analysis
      - Correlation studies
   
   d. Technical Analysis:
      - Indicator calculations
      - Pattern recognition
      - Support/resistance levels
      - Trading signals

5. Response Types:
   - Text responses with formatting
   - Data tables
   - Interactive charts
   - Alert notifications
   - Error messages
   - Help suggestions

6. Integration Features:
   - Real-time market data
   - Portfolio management
   - Watchlist tracking
   - News aggregation
   - Technical analysis tools

7. Error Handling:
   - Input validation
   - API error handling
   - Data validation
   - Fallback responses
   - User feedback collection

8. Performance Optimization:
   - Response caching
   - Query optimization
   - Resource management
   - Load balancing
   - Rate limiting

### 3. Technical Features

#### 3.1 Data Integration
- Yahoo Finance API integration for real-time data
- Historical data analysis
- Market news aggregation
- Technical indicator calculations

#### 3.2 AI/ML Capabilities
- Natural Language Processing for chat interface
- Sentiment analysis using transformer models
- Price prediction models
- Technical analysis algorithms

#### 3.3 User Interface
- Modern dark theme
- Responsive design
- Interactive charts and graphs
- Real-time data updates
- Intuitive navigation

#### 3.4 Data Management
- Portfolio data persistence
- Watchlist management
- Transaction history
- User preferences

### 4. Security Features
- Secure data storage
- Input validation
- Error handling
- Data backup mechanisms

### 5. Future Enhancements
- Additional technical indicators
- Enhanced AI predictions
- More market data sources
- Advanced portfolio analytics
- Mobile responsiveness improvements

### 6. Technical Requirements
- Python 3.x
- Streamlit
- Yahoo Finance API
- Pandas
- Plotly
- TensorFlow/PyTorch
- Other dependencies as specified in requirements.txt

### 7. Installation and Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

### 8. Usage Guidelines
1. Start with the Dashboard for market overview
2. Use Market Activity for detailed analysis
3. Manage your portfolio and watchlist
4. Utilize AI tools for advanced insights
5. Interact with the AI Chat for quick information

### 9. Support and Maintenance
- Regular updates for market data integration
- Bug fixes and performance improvements
- New feature additions
- User feedback incorporation

This documentation provides a comprehensive overview of the StockRaj platform. For specific implementation details, refer to the individual component documentation and code comments. 