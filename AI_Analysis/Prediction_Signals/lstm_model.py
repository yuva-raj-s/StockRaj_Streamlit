import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_lstm_model():
    """Create and return a simple LSTM model with reduced memory usage"""
    try:
        model = Sequential([
            LSTM(16, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(16, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    except Exception as e:
        logger.error(f"Error creating LSTM model: {str(e)}")
        return None

def prepare_data_for_lstm(data, sequence_length=60):
    """Prepare data for LSTM prediction"""
    try:
        # Ensure we have at least 3 months of data
        if len(data) < 63:  # 63 trading days in 3 months
            raise ValueError("Insufficient data for training. Need at least 3 months of historical data.")
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        # Prepare sequences
        X = []
        y = []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and validation
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        return X_train, y_train, X_val, y_val, scaler
    except Exception as e:
        logger.error(f"Error preparing data for LSTM: {str(e)}")
        return None, None, None, None, None

def predict_future_prices(data, days=15):
    """Predict future prices using LSTM model with reduced memory usage"""
    try:
        # Load the LSTM model
        model = create_lstm_model()
        if model is None:
            logger.error("Failed to create LSTM model")
            return None, None, None
            
        # Prepare data for prediction
        X_train, y_train, X_val, y_val, scaler = prepare_data_for_lstm(data)
        if X_train is None:
            logger.error("Failed to prepare data for LSTM")
            return None, None, None
            
        # Train the model
        try:
            model.fit(X_train, y_train, 
                     validation_data=(X_val, y_val),
                     epochs=8, 
                     batch_size=8,
                     verbose=0)
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None, None, None
            
        # Get the last sequence for prediction
        last_sequence = X_val[-1].reshape(1, 60, 1)
        
        # Make predictions for next 15 days
        predictions = []
        confidence_scores = []
        signals = []
        current_sequence = last_sequence
        
        for _ in range(days):
            try:
                next_pred = model.predict(current_sequence, verbose=0)
                pred_price = scaler.inverse_transform(next_pred)[0, 0]
                predictions.append(pred_price)
                
                # Calculate confidence score based on prediction stability
                confidence = min(0.99, max(0.1, 1 - abs(next_pred[0, 0] - current_sequence[0, -1, 0])))
                confidence_scores.append(confidence)
                
                # Generate signal based on prediction
                if pred_price > scaler.inverse_transform(current_sequence[0, -1, 0].reshape(-1, 1))[0, 0]:
                    signals.append("BUY")
                elif pred_price < scaler.inverse_transform(current_sequence[0, -1, 0].reshape(-1, 1))[0, 0]:
                    signals.append("SELL")
                else:
                    signals.append("HOLD")
                    
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[0, -1, 0] = next_pred[0, 0]
            except Exception as e:
                logger.error(f"Error in prediction step: {str(e)}")
                tf.keras.backend.clear_session()
                return None, None, None
                
        tf.keras.backend.clear_session()
        return predictions, confidence_scores, signals
    except Exception as e:
        logger.error(f"Error in future price prediction: {str(e)}")
        tf.keras.backend.clear_session()
        return None, None, None 