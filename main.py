from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import Dict, Any
from data.prices import get_price
from data.indicators import get_technical_indicators
from nlp.news import get_recent_headlines
from ml.predict import predict_price
from ml.train import train_model
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Market Analysis API",
    description="API for analyzing stock market data",
    version="1.0.0"
)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Welcome to the Stock Market Analysis API"}

@app.post("/train/{ticker}")
async def train_stock_model(ticker: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Train the LSTM model for a specific stock ticker.
    Training is done in the background to not block the API.
    """
    try:
        # Start training in background
        background_tasks.add_task(train_model, ticker)
        
        return {
            "message": f"Started training model for {ticker}",
            "status": "training"
        }
    except Exception as e:
        logger.error(f"Error starting training for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/{ticker}")
async def analyze_stock(ticker: str) -> Dict[str, Any]:
    """
    Analyze a stock ticker and return analysis results
    """
    logger.info(f"Starting analysis for ticker: {ticker}")
    
    # Get real price data
    logger.info(f"Fetching current price for {ticker}")
    current_price = get_price(ticker)
    
    if current_price is None:
        logger.error(f"Failed to fetch price for {ticker}")
        raise HTTPException(status_code=404, detail=f"Could not fetch data for ticker {ticker}")
    
    logger.info(f"Successfully fetched price for {ticker}: ${current_price:.2f}")
    
    # Get technical indicators
    logger.info(f"Computing technical indicators for {ticker}")
    indicators = get_technical_indicators(ticker)
    rsi_val = indicators.get("rsi")
    ma_val = indicators.get("ma_20")
    volume = indicators.get("volume", 0)

    if rsi_val is not None and ma_val is not None:
        logger.info(f"RSI: {rsi_val:.2f}, MA20: {ma_val:.2f}")
    else:
        logger.warning(f"Could not compute technical indicators. RSI: {rsi_val}, MA20: {ma_val}")
    
    # Get news headlines with sentiment
    logger.info(f"Analyzing sentiment for {ticker}")
    headlines = get_recent_headlines(ticker)
    
    # Calculate overall sentiment score
    technical_sentiment = 1.0 if rsi_val and rsi_val > 50 else 0.0
    news_sentiment = headlines[0]["sentiment_score"] if headlines else 0.5
    overall_sentiment = (technical_sentiment + news_sentiment) / 2
    logger.info(f"Overall sentiment score: {overall_sentiment:.2f}")
    
    # Prepare features for prediction
    prediction_features = {
        "ticker": ticker,
        "current_price": current_price,
        "rsi": rsi_val,
        "ma_20": ma_val,
        "sentiment_score": overall_sentiment,
        "volume": volume
    }
    
    # Get price prediction
    logger.info(f"Generating price prediction for {ticker}")
    predicted_price = predict_price(prediction_features)
    if predicted_price:
        logger.info(f"Predicted price: ${predicted_price:.2f}")
    else:
        logger.warning(f"Failed to generate price prediction for {ticker}")
    
    # Round all float values to 2 decimal places
    analysis_result = {
        "ticker": ticker,
        "price": round(current_price, 2),
        "rsi": round(rsi_val, 2) if rsi_val is not None else None,
        "moving_average_20": round(ma_val, 2) if ma_val is not None else None,
        "sentiment_score": round(overall_sentiment, 2),
        "predicted_price": round(predicted_price, 2) if predicted_price is not None else None,
        "model_status": "trained" if os.path.exists(f"models/{ticker}_lstm.pth") else "untrained",
        "news_articles": headlines  # Include the news articles in the response
    }
    
    logger.info(f"Completed analysis for {ticker}")
    return analysis_result 