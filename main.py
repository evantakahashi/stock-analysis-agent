from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from data.prices import get_price

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
    # Get real price data
    current_price = get_price(ticker)
    
    if current_price is None:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for ticker {ticker}")
    
    # Dummy data for demonstration (keeping other fields as dummy data for now)
    analysis_result = {
        "ticker": ticker,
        "current_price": current_price,
        "price_change": 2.5,  # This could be calculated from historical data
        "volume": 1000000,
        "market_cap": "1.5B",
        "analysis": {
            "sentiment": "positive",
            "trend": "bullish",
            "confidence_score": 0.85
        }
    }
    
    return analysis_result 