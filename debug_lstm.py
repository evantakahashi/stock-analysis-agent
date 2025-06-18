import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from ml.lstm_model import StockPricePredictor
from data.prices import get_price, get_intraday_data, get_historical_data
from data.indicators import get_technical_indicators
from nlp.news import get_recent_headlines

def analyze_model_predictions(ticker: str = "AAPL"):
    """
    Comprehensive analysis of LSTM model predictions for debugging.
    """
    print(f"🔍 Debugging LSTM Model for {ticker}")
    print("=" * 50)
    
    # 1. Check if model exists
    model_path = f'models/{ticker}_lstm.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please train the model first using: python -m ml.train --ticker AAPL")
        return
    
    print(f"✅ Model found at {model_path}")
    
    # 2. Load current market data
    print("\n📊 Fetching current market data...")
    current_price = get_price(ticker)
    if current_price is None:
        print(f"❌ Failed to fetch current price for {ticker}")
        return
    
    print(f"   Current price: ${current_price:.2f}")
    
    # 3. Get technical indicators
    indicators = get_technical_indicators(ticker)
    rsi = indicators.get('rsi')
    ma_20 = indicators.get('ma_20')
    volume = indicators.get('volume', current_price)
    
    print(f"   RSI: {rsi:.2f}" if rsi else "   RSI: N/A")
    print(f"   MA20: ${ma_20:.2f}" if ma_20 else "   MA20: N/A")
    print(f"   Volume: {volume:,.0f}")
    
    # 4. Get news sentiment
    headlines = get_recent_headlines(ticker)
    sentiment = headlines[0]['sentiment_score'] if headlines else 0.5
    print(f"   Sentiment Score: {sentiment:.2f}")
    
    # 5. Load and analyze the model
    print(f"\n🤖 Loading LSTM model...")
    predictor = StockPricePredictor()
    
    try:
        predictor.model.load_state_dict(torch.load(model_path, weights_only=True))
        predictor.model.eval()
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading model: {str(e)}")
        return
    
    # 6. Analyze model architecture
    print(f"\n🏗️  Model Architecture Analysis:")
    total_params = sum(p.numel() for p in predictor.model.parameters())
    trainable_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # 7. Test prediction with current data
    print(f"\n🔮 Testing prediction with current data...")
    
    # Prepare prediction data (this is where the issue might be)
    prediction_data = {
        'prices': [current_price] * 60,  # This is problematic - using same price 60 times
        'volume': [volume] * 60,
        'rsi': [rsi if rsi else 50] * 60,
        'ma': [ma_20 if ma_20 else current_price] * 60,
        'sentiment': [sentiment] * 60,
        'open': [current_price] * 60,
        'high': [current_price] * 60,
        'low': [current_price] * 60
    }
    
    # Make prediction
    with torch.no_grad():
        predicted_price = predictor.predict(prediction_data)
    
    print(f"   Predicted price: ${predicted_price:.2f}")
    print(f"   Price difference: ${predicted_price - current_price:.2f}")
    print(f"   Percentage change: {((predicted_price - current_price) / current_price) * 100:.2f}%")
    
    # 8. Analyze the issue - the problem is in the data preparation
    print(f"\n🚨 IDENTIFIED ISSUES:")
    print(f"   1. Prediction data uses the same values repeated 60 times")
    print(f"      - This creates unrealistic sequences for the LSTM")
    print(f"      - LSTM expects temporal patterns, not static values")
    print(f"   2. No historical context is being used")
    print(f"   3. The model was trained on intraday data but prediction uses static data")
    
    # 9. Get historical data for better analysis
    print(f"\n📈 Fetching historical data for analysis...")
    historical_data = get_historical_data(ticker, days=90)
    
    if historical_data:
        print(f"   ✅ Retrieved {len(historical_data['close'])} days of historical data")
        
        # Analyze price range
        min_price = min(historical_data['close'])
        max_price = max(historical_data['close'])
        avg_price = np.mean(historical_data['close'])
        
        print(f"   Price range: ${min_price:.2f} - ${max_price:.2f}")
        print(f"   Average price: ${avg_price:.2f}")
        print(f"   Current price vs range: {((current_price - min_price) / (max_price - min_price)) * 100:.1f}%")
        
        # Check if prediction is reasonable
        if predicted_price < min_price * 0.5 or predicted_price > max_price * 1.5:
            print(f"   ⚠️  WARNING: Prediction (${predicted_price:.2f}) is outside reasonable range!")
        else:
            print(f"   ✅ Prediction is within reasonable range")
    
    # 10. Suggest fixes
    print(f"\n🔧 RECOMMENDED FIXES:")
    print(f"   1. Use actual historical sequences for prediction, not repeated values")
    print(f"   2. Implement proper data preprocessing in predict.py")
    print(f"   3. Add validation to ensure predictions are within reasonable bounds")
    print(f"   4. Consider using different normalization techniques")
    print(f"   5. Add more features like volatility, momentum indicators")
    
    return {
        'current_price': current_price,
        'predicted_price': predicted_price,
        'price_difference': predicted_price - current_price,
        'percentage_change': ((predicted_price - current_price) / current_price) * 100,
        'model_path': model_path,
        'total_params': total_params
    }

def create_fixed_prediction_function():
    """
    Create a fixed version of the prediction function.
    """
    print(f"\n🔧 Creating fixed prediction function...")
    
    fixed_code = '''
def predict_price_fixed(features: Dict[str, Any]) -> Optional[float]:
    """
    Fixed version of predict_price that uses proper historical data.
    """
    try:
        ticker = features.get("ticker", "AAPL")
        
        # Initialize predictor
        predictor = StockPricePredictor()
        
        # Check if model exists
        model_path = f'models/{ticker}_lstm.pth'
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Using fallback prediction.")
            return features.get('current_price', 0) + 1.5
        
        # Load model
        predictor.model.load_state_dict(torch.load(model_path, weights_only=True))
        predictor.model.eval()
        
        # Get historical data for proper sequence
        historical_data = get_historical_data(ticker, days=90)
        if not historical_data:
            print("Failed to fetch historical data")
            return features.get('current_price', 0)
        
        # Get technical indicators for the sequence
        indicators = get_technical_indicators(ticker)
        
        # Prepare proper sequence data
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
        
        # Make prediction
        with torch.no_grad():
            predicted_price = predictor.predict(recent_data)
            
        # Add validation
        current_price = features.get('current_price', historical_data['close'][-1])
        price_range = max(historical_data['close']) - min(historical_data['close'])
        min_reasonable = min(historical_data['close']) - price_range * 0.1
        max_reasonable = max(historical_data['close']) + price_range * 0.1
        
        if predicted_price < min_reasonable or predicted_price > max_reasonable:
            print(f"Warning: Prediction {predicted_price:.2f} outside reasonable range [{min_reasonable:.2f}, {max_reasonable:.2f}]")
            # Use a more conservative prediction
            predicted_price = current_price + (predicted_price - current_price) * 0.1
            
        return predicted_price
        
    except Exception as e:
        print(f"Error in price prediction: {str(e)}")
        return None
'''
    
    print("   ✅ Fixed prediction function created")
    print("   📝 Copy this function to replace the current predict_price function in ml/predict.py")
    
    return fixed_code

if __name__ == "__main__":
    # Run the analysis
    results = analyze_model_predictions("AAPL")
    
    if results:
        # Create the fixed function
        fixed_function = create_fixed_prediction_function()
        
        # Save results to file
        with open('debug_results.txt', 'w') as f:
            f.write("LSTM Model Debug Results\\n")
            f.write("=" * 30 + "\\n")
            f.write(f"Current Price: ${results['current_price']:.2f}\\n")
            f.write(f"Predicted Price: ${results['predicted_price']:.2f}\\n")
            f.write(f"Price Difference: ${results['price_difference']:.2f}\\n")
            f.write(f"Percentage Change: {results['percentage_change']:.2f}%\\n")
            f.write(f"Model Parameters: {results['total_params']:,}\\n")
        
        print(f"\\n📄 Results saved to debug_results.txt")
        print(f"🎯 Next steps: Implement the fixed prediction function") 