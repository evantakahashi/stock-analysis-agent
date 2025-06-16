from typing import List, Dict
from datetime import datetime, timedelta
from transformers import pipeline
import numpy as np

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(headlines: List[str]) -> float:
    """
    Analyze the sentiment of a list of headlines using HuggingFace transformers.
    
    Args:
        headlines (List[str]): List of news headlines to analyze
        
    Returns:
        float: Average confidence score for positive sentiment (0-1)
    """
    try:
        # Get sentiment analysis for each headline
        results = sentiment_analyzer(headlines)
        
        # Calculate average confidence for positive sentiment
        positive_scores = []
        for result in results:
            if result['label'] == 'POSITIVE':
                positive_scores.append(result['score'])
            else:
                # For negative sentiment, use (1 - score) to get positive confidence
                positive_scores.append(1 - result['score'])
        
        return float(np.mean(positive_scores)) if positive_scores else 0.5
        
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return 0.5  # Return neutral sentiment on error

def get_recent_headlines(ticker: str) -> List[Dict[str, str]]:
    """
    Get recent news headlines for a given ticker.
    Currently returns placeholder data.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        List of dictionaries containing headline data with title, source, and date
    """
    # Get current date
    today = datetime.now()
    
    # Placeholder headlines
    headlines = [
        {
            "title": f"{ticker} Announces Major Product Launch, Stock Surges",
            "source": "Financial Times",
            "date": (today - timedelta(days=1)).strftime("%Y-%m-%d")
        },
        {
            "title": f"Analysts Upgrade {ticker} Price Target Following Strong Earnings",
            "source": "Bloomberg",
            "date": (today - timedelta(days=2)).strftime("%Y-%m-%d")
        },
        {
            "title": f"{ticker} Expands Market Share in Key Growth Segment",
            "source": "Wall Street Journal",
            "date": (today - timedelta(days=3)).strftime("%Y-%m-%d")
        }
    ]
    
    # Extract just the titles for sentiment analysis
    titles = [headline["title"] for headline in headlines]
    
    # Add sentiment score to each headline
    sentiment_score = analyze_sentiment(titles)
    for headline in headlines:
        headline["sentiment_score"] = sentiment_score
    
    return headlines 