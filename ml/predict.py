from typing import Dict, Any, Optional

def predict_price(features: Dict[str, Any]) -> Optional[float]:
    """
    Predict the future price based on current features.
    Currently returns a dummy prediction (current price + 1.5).
    
    Args:
        features (Dict[str, Any]): Dictionary containing:
            - current_price: float
            - rsi: Optional[float]
            - ma_20: Optional[float]
            - sentiment_score: float
            
    Returns:
        Optional[float]: Predicted price, or None if prediction fails
    """
    try:
        # Extract current price from features
        current_price = features.get('current_price')
        
        if current_price is None:
            return None
            
        # Dummy prediction: current price + 1.5
        predicted_price = current_price + 1.5
        
        return predicted_price
        
    except Exception as e:
        print(f"Error in price prediction: {str(e)}")
        return None 