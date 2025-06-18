from typing import List, Dict
from datetime import datetime, timedelta
from transformers import pipeline
import numpy as np
import os
import finnhub

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))

def analyze_sentiment(headlines: List[str]) -> float:
    """
    Analyze the sentiment of a list of headlines using HuggingFace transformers.
    Handles empty headlines and provides more robust sentiment analysis.
    
    Args:
        headlines (List[str]): List of news headlines to analyze
        
    Returns:
        float: Average confidence score for positive sentiment (0-1)
    """
    if not headlines:
        return 0.5  # Return neutral sentiment for empty headlines
        
    try:
        # Clean headlines by removing special characters and extra whitespace
        cleaned_headlines = [
            ' '.join(headline.split())  # Normalize whitespace
            for headline in headlines
        ]
        
        # Get sentiment analysis for each headline
        results = sentiment_analyzer(cleaned_headlines)
        
        # Calculate weighted average confidence for positive sentiment
        positive_scores = []
        for result in results:
            if result['label'] == 'POSITIVE':
                positive_scores.append(result['score'])
            else:
                # For negative sentiment, use (1 - score) to get positive confidence
                positive_scores.append(1 - result['score'])
        
        # Calculate average with confidence weighting
        if positive_scores:
            # Weight recent headlines more heavily
            weights = np.linspace(1.0, 0.8, len(positive_scores))
            weighted_scores = np.average(positive_scores, weights=weights)
            return float(weighted_scores)
        
        return 0.5  # Return neutral sentiment if no valid scores
        
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return 0.5  # Return neutral sentiment on error

def get_recent_headlines(ticker: str) -> List[Dict[str, str]]:
    """
    Get recent news headlines for a given ticker using Finnhub.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        List of dictionaries containing headline data with title, source, date, and url
    """
    try:
        # Get company news from Finnhub
        news = finnhub_client.company_news(ticker, _from=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                                         to=datetime.now().strftime('%Y-%m-%d'))
        
        # Process the articles
        headlines = []
        for article in news[:5]:  # Get top 5 most recent articles
            headline = {
                "title": article['headline'],
                "source": article['source'],
                "date": article['datetime'],
                "url": article['url'],
                "category": article.get('category', 'General'),
                "summary": article.get('summary', '')
            }
            headlines.append(headline)
        
        # Extract just the titles for sentiment analysis
        titles = [headline["title"] for headline in headlines]
        
        # Add sentiment score to each headline
        sentiment_score = analyze_sentiment(titles)
        for headline in headlines:
            headline["sentiment_score"] = sentiment_score
        
        return headlines
        
    except Exception as e:
        print(f"Error fetching news headlines: {str(e)}")
        return [] 