import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
from .lstm_model import StockPricePredictor
from data.prices import get_price, get_intraday_data, get_historical_data
from data.indicators import get_technical_indicators
from nlp.news import get_recent_headlines
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_training_data(ticker: str, lookback_days: int = 90) -> Dict[str, List[float]]:
    """
    Prepare training data for the model using intraday data.
    
    Args:
        ticker: Stock ticker symbol
        lookback_days: Number of historical days to use
        
    Returns:
        Dictionary containing prepared training data
    """
    try:
        # Get intraday data (60-minute intervals)
        intraday_data = get_intraday_data(ticker, interval='60min', days=lookback_days)
        if not intraday_data:
            logger.error(f"Failed to fetch intraday data for {ticker}")
            return None
            
        # Get technical indicators
        indicators = get_technical_indicators(ticker)
        rsi_values = [indicators.get('rsi', 50)] * len(intraday_data['close'])
        ma_values = [indicators.get('ma_20', intraday_data['close'][0])] * len(intraday_data['close'])
        
        # Get news sentiment
        headlines = get_recent_headlines(ticker)
        sentiment = headlines[0]['sentiment_score'] if headlines else 0.5
        sentiment_scores = [sentiment] * len(intraday_data['close'])
        
        # Ensure all required features are present
        return {
            'prices': intraday_data['close'],
            'volume': intraday_data['volume'],
            'rsi': rsi_values,
            'ma': ma_values,
            'sentiment': sentiment_scores,
            'open': intraday_data['open'],
            'high': intraday_data['high'],
            'low': intraday_data['low']
        }
        
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        return None

def train_model(ticker: str, epochs: int = 300, batch_size: int = 32):
    """
    Train the LSTM model on historical data.
    
    Args:
        ticker: Stock ticker symbol
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    try:
        # Initialize model with more input features
        predictor = StockPricePredictor(input_dim=8)  # [close, volume, rsi, ma, sentiment, open, high, low]
        
        # Prepare training data
        data = prepare_training_data(ticker)
        if not data:
            logger.error("Failed to prepare training data")
            return
        
        # Convert data to tensors
        prices = torch.FloatTensor(data['prices']).to(predictor.device)
        
        # Training loop
        logger.info(f"Starting training for {ticker}")
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Create batches
            for i in range(0, len(prices) - batch_size, batch_size):
                # Prepare sequence
                sequence_data = {
                    'prices': data['prices'][i:i+batch_size],
                    'volume': data['volume'][i:i+batch_size],
                    'rsi': data['rsi'][i:i+batch_size],
                    'ma': data['ma'][i:i+batch_size],
                    'sentiment': data['sentiment'][i:i+batch_size],
                    'open': data['open'][i:i+batch_size],
                    'high': data['high'][i:i+batch_size],
                    'low': data['low'][i:i+batch_size]
                }
                
                # Get target (next hour's price)
                target = prices[i+1:i+batch_size+1]
                
                # Training step
                loss = predictor.train_step(
                    predictor.prepare_sequence(sequence_data),
                    target
                )
                
                total_loss += loss
                num_batches += 1
            
            # Log progress
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        torch.save(predictor.model.state_dict(), f'models/{ticker}_lstm.pth')
        logger.info(f"Model saved for {ticker}")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train LSTM model for stock price prediction')
    parser.add_argument('--ticker', type=str, default='AAPL',
                      help='Stock ticker symbol to train on (default: AAPL)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--lookback-days', type=int, default=90,
                      help='Number of historical days to use (default: 90)')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train model
    logger.info(f"Training model for {args.ticker}")
    train_model(
        ticker=args.ticker,
        epochs=args.epochs,
        batch_size=args.batch_size
    ) 