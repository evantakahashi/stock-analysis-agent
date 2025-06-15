from alpha_vantage.timeseries import TimeSeries
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if not API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables")

# Initialize TimeSeries
ts = TimeSeries(key=API_KEY, output_format='pandas')

def get_price(ticker: str) -> Optional[float]:
    """
    Fetch the most recent closing price for a given ticker symbol using Alpha Vantage.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        Optional[float]: The most recent closing price, or None if the ticker is invalid
    """
    try:
        # Get the most recent data
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='compact')
        
        if data.empty:
            return None
            
        # Return the most recent closing price
        return float(data['4. close'].iloc[0])
        
    except Exception as e:
        print(f"Error fetching price for {ticker}: {str(e)}")
        return None 