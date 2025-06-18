from alpha_vantage.timeseries import TimeSeries
from typing import Optional, Dict, List
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta

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

def get_intraday_data(ticker: str, interval: str = '60min', days: int = 30) -> Optional[Dict[str, List[float]]]:
    """
    Fetch intraday OHLCV data for a given ticker using Alpha Vantage.
    
    Args:
        ticker (str): The stock ticker symbol
        interval (str): Time interval between data points ('1min', '5min', '15min', '30min', '60min')
        days (int): Number of days of historical data to fetch
        
    Returns:
        Optional[Dict[str, List[float]]]: Dictionary containing lists of OHLCV data, or None if fetch fails
    """
    try:
        # Get intraday data
        data, meta_data = ts.get_intraday(
            symbol=ticker,
            interval=interval,
            outputsize='full' if days > 30 else 'compact'
        )
        
        if data.empty:
            return None
            
        # Rename columns for easier access
        data.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        }, inplace=True)
        
        # Calculate number of data points needed
        points_needed = days * 24 // int(interval[:-3])
        
        # Get the most recent data points
        recent_data = data.iloc[-points_needed:]
        
        # Convert to lists
        result = {
            'open': recent_data['open'].tolist(),
            'high': recent_data['high'].tolist(),
            'low': recent_data['low'].tolist(),
            'close': recent_data['close'].tolist(),
            'volume': recent_data['volume'].tolist(),
            'timestamps': recent_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        }
        
        return result
        
    except Exception as e:
        print(f"Error fetching intraday data for {ticker}: {str(e)}")
        return None

def get_historical_data(ticker: str, days: int = 90) -> Optional[Dict[str, List[float]]]:
    """
    Fetch historical daily data for a given ticker.
    
    Args:
        ticker (str): The stock ticker symbol
        days (int): Number of days of historical data to fetch
        
    Returns:
        Optional[Dict[str, List[float]]]: Dictionary containing lists of OHLCV data, or None if fetch fails
    """
    try:
        # Get daily data
        data, meta_data = ts.get_daily(
            symbol=ticker,
            outputsize='full' if days > 100 else 'compact'
        )
        
        if data.empty:
            return None
            
        # Rename columns for easier access
        data.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        }, inplace=True)
        
        # Convert to lists and get the most recent 'days' worth of data
        result = {
            'open': data['open'].iloc[-days:].tolist(),
            'high': data['high'].iloc[-days:].tolist(),
            'low': data['low'].iloc[-days:].tolist(),
            'close': data['close'].iloc[-days:].tolist(),
            'volume': data['volume'].iloc[-days:].tolist(),
            'timestamps': data.index[-days:].tolist()
        }
        
        return result
        
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {str(e)}")
        return None 