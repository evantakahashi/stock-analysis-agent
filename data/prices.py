import yfinance as yf
from typing import Optional

def get_price(ticker: str) -> Optional[float]:
    """
    Fetch the most recent closing price for a given ticker symbol.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        Optional[float]: The most recent closing price, or None if the ticker is invalid
    """
    try:
        # Create a Ticker object
        stock = yf.Ticker(ticker)
        
        # Get the most recent data (1 day)
        hist = stock.history(period='1d')
        
        if hist.empty:
            return None
            
        # Return the most recent closing price
        return float(hist['Close'].iloc[-1])
        
    except Exception as e:
        print(f"Error fetching price for {ticker}: {str(e)}")
        return None 