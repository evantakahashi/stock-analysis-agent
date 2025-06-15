from fastapi import FastAPI, HTTPException
from typing import Dict, Any

app = FastAPI(
    title="Stock Market Analysis API",
    description="API for analyzing stock market data",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Stock Market Analysis API"}

@app.get("/analyze/{ticker}")
async def analyze_stock(ticker: str) -> Dict[str, Any]:
    """
    Analyze a stock ticker and return analysis results
    """
    # Dummy data for demonstration
    analysis_result = {
        "ticker": ticker,
        "current_price": 150.25,
        "price_change": 2.5,
        "volume": 1000000,
        "market_cap": "1.5B",
        "analysis": {
            "sentiment": "positive",
            "trend": "bullish",
            "confidence_score": 0.85
        }
    }
    
    return analysis_result 