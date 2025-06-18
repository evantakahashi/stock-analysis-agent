import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import time

# Configure the page
st.set_page_config(
    page_title="Stock Market Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Stock Market Analysis Dashboard")
st.markdown("Enter a stock ticker to analyze its current market position and predictions.")

# Input for stock ticker
ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()

# Function to fetch data from FastAPI
def fetch_stock_data(ticker):
    try:
        response = requests.get(f"http://localhost:8000/analyze/{ticker}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to train model
def train_model(ticker):
    try:
        response = requests.post(f"http://localhost:8000/train/{ticker}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error starting training: {str(e)}")
        return None

# Create columns for controls
control_col1, control_col2 = st.columns([1, 3])

if ticker:
    with control_col1:
        # Add train button
        if st.button("Train Model"):
            with st.spinner("Starting model training..."):
                result = train_model(ticker)
                if result:
                    st.success(result["message"])
                    # Start polling for updates
                    for _ in range(30):  # Poll for up to 30 seconds
                        time.sleep(1)
                        data = fetch_stock_data(ticker)
                        if data and data.get("model_status") == "trained":
                            st.success("Model training completed!")
                            st.rerun()
                            break
    
    with control_col2:
        # Add refresh button
        if st.button("Refresh Data"):
            st.rerun()
    
    data = fetch_stock_data(ticker)
    
    if data:
        # Display model status
        model_status = data.get("model_status", "unknown")
        if model_status == "untrained":
            st.warning("Model not trained for this stock. Click 'Train Model' to start training.")
        else:
            st.success("Model is trained and ready for predictions.")
        
        # Create two columns for metrics
        col1, col2 = st.columns(2)
        
        # Display metrics in columns
        with col1:
            st.metric("Current Price", f"${data['price']:.2f}")
            st.metric("RSI", f"{data['rsi']:.2f}" if data['rsi'] else "N/A")
            st.metric("20-Day Moving Average", f"${data['moving_average_20']:.2f}" if data['moving_average_20'] else "N/A")
        
        with col2:
            st.metric("Sentiment Score", f"{data['sentiment_score']:.2f}")
            st.metric("Predicted Price", f"${data['predicted_price']:.2f}" if data['predicted_price'] else "N/A")
        
        # Create price comparison chart
        if data['price'] and data['predicted_price']:
            # Create a simple line chart comparing current and predicted prices
            fig = go.Figure()
            
            # Add current price
            fig.add_trace(go.Scatter(
                x=['Current'],
                y=[data['price']],
                mode='markers+lines',
                name='Current Price',
                line=dict(color='blue', width=2)
            ))
            
            # Add predicted price
            fig.add_trace(go.Scatter(
                x=['Predicted'],
                y=[data['predicted_price']],
                mode='markers+lines',
                name='Predicted Price',
                line=dict(color='green', width=2)
            ))
            
            # Update layout
            fig.update_layout(
                title='Price Comparison',
                xaxis_title='',
                yaxis_title='Price ($)',
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display price difference
            price_diff = data['predicted_price'] - data['price']
            price_change_pct = (price_diff / data['price']) * 100
            
            st.markdown(f"""
            ### Price Analysis
            - **Price Difference**: ${price_diff:.2f}
            - **Percentage Change**: {price_change_pct:.2f}%
            """)
            
            # Add interpretation
            if price_diff > 0:
                st.success(f"The model predicts a potential increase of {price_change_pct:.2f}% in {ticker}'s price.")
            else:
                st.warning(f"The model predicts a potential decrease of {abs(price_change_pct):.2f}% in {ticker}'s price.")
        
        # Display news articles
        if 'news_articles' in data and data['news_articles']:
            st.markdown("### Recent News Articles")
            for article in data['news_articles']:
                with st.container():
                    st.markdown(f"""
                    #### [{article['title']}]({article['url']})
                    - **Source**: {article['source']}
                    - **Date**: {article['date']}
                    - **Category**: {article['category']}
                    - **Sentiment Score**: {article['sentiment_score']:.2f}
                    """)
                    if article.get('summary'):
                        st.markdown(f"**Summary**: {article['summary']}")
                    st.markdown("---") 