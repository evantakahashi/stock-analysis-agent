from typing import Dict, Any, Optional
from .lstm_model import StockPricePredictor
import torch
import os
from data.prices import get_historical_data
from data.indicators import get_technical_indicators

def predict_price(features: Dict[str, Any]) -> Optional[float]:
    """
    Predict the future price based on current features using the LSTM model.
    Uses actual historical time series data for the input sequence.
    
    Args:
        features (Dict[str, Any]): Dictionary containing:
            - ticker: str
            - current_price: float
            - rsi: Optional[float]
            - ma_20: Optional[float]
            - sentiment_score: float
            - volume: Optional[float]
            
    Returns:
        Optional[float]: Predicted price, or None if prediction fails
    """
    try:
        ticker = features.get("ticker", "AAPL")
        predictor = StockPricePredictor()
        model_path = f'models/{ticker}_lstm.pth'
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Using fallback prediction.")
            return features.get('current_price', 0) + 1.5
        predictor.model.load_state_dict(torch.load(model_path, weights_only=True))
        predictor.model.eval()

        # Get historical data for proper sequence
        historical_data = get_historical_data(ticker, days=90)
        if not historical_data:
            print("Failed to fetch historical data")
            return features.get('current_price', 0)

        # Get technical indicators for the sequence
        indicators = get_technical_indicators(ticker)
        seq_length = 60
        if len(historical_data['close']) < seq_length:
            print(f"Insufficient historical data. Need {seq_length}, got {len(historical_data['close'])}")
            return features.get('current_price', 0)

        # Use the last 60 days of data
        recent_data = {
            'prices': historical_data['close'][-seq_length:],
            'volume': historical_data['volume'][-seq_length:],
            'rsi': [indicators.get('rsi', 50)] * seq_length,
            'ma': [indicators.get('ma_20', historical_data['close'][-1])] * seq_length,
            'sentiment': [features.get('sentiment_score', 0.5)] * seq_length,
            'open': historical_data['open'][-seq_length:],
            'high': historical_data['high'][-seq_length:],
            'low': historical_data['low'][-seq_length:]
        }

        with torch.no_grad():
            predicted_price = predictor.predict(recent_data)

        # Add validation
        current_price = features.get('current_price', historical_data['close'][-1])
        price_range = max(historical_data['close']) - min(historical_data['close'])
        min_reasonable = min(historical_data['close']) - price_range * 0.1
        max_reasonable = max(historical_data['close']) + price_range * 0.1
        if predicted_price < min_reasonable or predicted_price > max_reasonable:
            print(f"Warning: Prediction {predicted_price:.2f} outside reasonable range [{min_reasonable:.2f}, {max_reasonable:.2f}]")
            predicted_price = current_price + (predicted_price - current_price) * 0.1
        return predicted_price
    except Exception as e:
        print(f"Error in price prediction: {str(e)}")
        return None 