import pandas_ta as ta
from alpha_vantage.timeseries import TimeSeries
import os
from dotenv import load_dotenv
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if not API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables")

# Initialize TimeSeries
ts = TimeSeries(key=API_KEY, output_format='pandas')

def get_technical_indicators(ticker: str) -> Dict[str, Optional[float]]:
    """
    Calculate RSI and 20-day moving average for a given ticker.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        Dict containing RSI and MA values, or None if calculation fails
    """
    try:
        # Get historical data (need more than 20 days for MA calculation)
        logger.info(f"Fetching daily data for {ticker}")
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
        
        if data.empty:
            logger.error(f"No data received for {ticker}")
            return {"rsi": None, "ma_20": None}
            
        # Log the data structure
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Available columns: {data.columns.tolist()}")
        logger.info(f"First few rows of data:\n{data.head()}")
        
        # Verify we have enough data points
        if len(data) < 20:
            logger.error(f"Insufficient data points for {ticker}. Need at least 20, got {len(data)}")
            return {"rsi": None, "ma_20": None}
        
        # Rename columns for easier access
        data.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        }, inplace=True)

        # Sort data by date in ascending order (oldest to newest)
        data = data.sort_index(ascending=True)

        # Calculate RSI (14 periods is standard)
        logger.info("Calculating RSI...")
        rsi = ta.rsi(data['close'], length=14)
        
        # Calculate 20-day moving average
        logger.info("Calculating 20-day MA...")
        ma_20 = ta.sma(data['close'], length=20)
        
        # Get the most recent non-NaN values (last row is the latest trading day)
        latest_rsi = rsi.dropna().iloc[-1] if not rsi.dropna().empty else None
        latest_ma = ma_20.dropna().iloc[-1] if not ma_20.dropna().empty else None
        
        logger.info(f"Calculated indicators for {ticker}:")
        logger.info(f"RSI: {latest_rsi}")
        logger.info(f"MA20: {latest_ma}")
        
        return {
            "rsi": float(latest_rsi) if latest_rsi is not None else None,
            "ma_20": float(latest_ma) if latest_ma is not None else None,
        }
        
    except Exception as e:
        logger.error(f"Error calculating indicators for {ticker}: {str(e)}")
        logger.exception("Full traceback:")
        return {"rsi": None, "ma_20": None} 