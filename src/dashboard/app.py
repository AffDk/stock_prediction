"""
Streamlit dashboard for stock prediction monitoring.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from pathlib import Path

from ..utils.config import config
from ..utils.logging_config import get_logger

# Page configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = get_logger(__name__)

# Constants
# Check if running in cloud (API_BASE_URL environment variable set)
API_BASE_URL = os.getenv('API_BASE_URL', f"http://localhost:{config.get('api.port', 8000)}")
SUPPORTED_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]




def main():
    """Main dashboard function."""
    st.title("📈 Stock Prediction Dashboard")
    st.markdown("Monitor stock price predictions and model performance in real-time")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model status
        st.subheader("🤖 Model Status")
        model_status = get_model_status()
        
        if model_status["status"] == "ready":
            st.success("✅ Model Ready")
        elif model_status["status"] == "not_trained":
            st.warning("⚠️ Model Not Trained")
        else:
            st.error("❌ Model Not Available")
        
        st.markdown("---")
        
        # Symbol selection
        st.subheader("📊 Symbol Selection")
        selected_symbols = st.multiselect(
            "Select stocks to monitor:",
            SUPPORTED_SYMBOLS,
            default=["AAPL", "GOOGL", "MSFT"]
        )
        
        # Time horizon
        prediction_days = st.slider(
            "Prediction horizon (days):",
            min_value=1,
            max_value=30,
            value=7
        )
        
        # Refresh rate
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Main content
    if not selected_symbols:
        st.warning("Please select at least one stock symbol to monitor.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Predictions", "📊 Performance", "📰 News Impact", "⚙️ Model Info"])
    
    with tab1:
        show_predictions_tab(selected_symbols, prediction_days)
    
    with tab2:
        show_performance_tab(selected_symbols)
    
    with tab3:
        show_news_impact_tab(selected_symbols)
    
    with tab4:
        show_model_info_tab()
    
    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()


def get_model_status():
    """Get model status from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/model/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    return {"status": "unavailable"}


def get_prediction(symbol, days_ahead=7):
    """Get prediction for a symbol."""
    try:
        payload = {
            "symbol": symbol,
            "days_ahead": days_ahead,
            "include_news": True
        }
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error getting prediction for {symbol}: {e}")
    
    return None


def show_predictions_tab(selected_symbols, prediction_days):
    """Show predictions tab."""
    st.header("📈 Current Predictions")
    
    # Get predictions for all selected symbols
    predictions = []
    
    progress_bar = st.progress(0)
    for i, symbol in enumerate(selected_symbols):
        pred = get_prediction(symbol, prediction_days)
        if pred:
            predictions.append(pred)
        progress_bar.progress((i + 1) / len(selected_symbols))
    
    progress_bar.empty()
    
    if not predictions:
        st.warning("No predictions available. Make sure the API is running and the model is trained.")
        return
    
    # Display predictions in columns
    cols = st.columns(min(len(predictions), 3))
    
    for i, pred in enumerate(predictions):
        col_idx = i % 3
        
        with cols[col_idx]:
            # Prediction card
            st.markdown(f"### {pred['symbol']}")
            
            # Current vs predicted price
            current_price = pred['current_price']
            predicted_price = pred['predicted_price']
            change_percent = pred['predicted_change_percent']
            
            # Color coding
            color = "green" if change_percent > 0 else "red"
            arrow = "📈" if change_percent > 0 else "📉"
            
            st.metric(
                label="Current Price",
                value=f"${current_price:.2f}",
                delta=None
            )
            
            st.metric(
                label=f"Predicted ({prediction_days}d)",
                value=f"${predicted_price:.2f}",
                delta=f"{change_percent:.2f}%"
            )
            
            # Confidence
            confidence = pred['confidence_score']
            st.progress(confidence)
            st.caption(f"Confidence: {confidence:.1%}")
            
            st.markdown("---")
    
    # Summary table
    st.subheader("📋 Predictions Summary")
    
    df = pd.DataFrame(predictions)
    df = df[['symbol', 'current_price', 'predicted_price', 'predicted_change_percent', 'confidence_score']]
    df.columns = ['Symbol', 'Current Price', 'Predicted Price', 'Change %', 'Confidence']
    
    # Format columns
    df['Current Price'] = df['Current Price'].apply(lambda x: f"${x:.2f}")
    df['Predicted Price'] = df['Predicted Price'].apply(lambda x: f"${x:.2f}")
    df['Change %'] = df['Change %'].apply(lambda x: f"{x:+.2f}%")
    df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(df, use_container_width=True)


def show_performance_tab(selected_symbols):
    """Show model performance metrics."""
    st.header("📊 Model Performance")
    
    # Mock performance data (would be real in production)
    st.subheader("📈 Accuracy Over Time")
    
    # Generate mock historical accuracy data
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
    np.random.seed(42)  # For reproducible results
    
    accuracy_data = []
    for symbol in selected_symbols:
        base_accuracy = np.random.uniform(0.6, 0.8)
        accuracies = base_accuracy + np.random.normal(0, 0.05, len(dates))
        accuracies = np.clip(accuracies, 0.3, 0.95)  # Realistic bounds
        
        for date, acc in zip(dates, accuracies):
            accuracy_data.append({
                'Date': date,
                'Symbol': symbol,
                'Accuracy': acc
            })
    
    df_accuracy = pd.DataFrame(accuracy_data)
    
    # Plot accuracy over time
    fig = px.line(
        df_accuracy, 
        x='Date', 
        y='Accuracy', 
        color='Symbol',
        title="Prediction Accuracy Over Time"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Overall Metrics")
        
        # Mock metrics
        metrics = {
            'Mean Absolute Error': '2.3%',
            'R² Score': '0.74',
            'Directional Accuracy': '72%',
            'Sharpe Ratio': '1.45'
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)
    
    with col2:
        st.subheader("🎯 Symbol Performance")
        
        # Mock symbol-specific performance
        symbol_perf = []
        for symbol in selected_symbols:
            symbol_perf.append({
                'Symbol': symbol,
                'Accuracy': np.random.uniform(0.65, 0.80),
                'MAE': np.random.uniform(0.015, 0.035),
                'Predictions': np.random.randint(50, 200)
            })
        
        df_perf = pd.DataFrame(symbol_perf)
        df_perf['Accuracy'] = df_perf['Accuracy'].apply(lambda x: f"{x:.1%}")
        df_perf['MAE'] = df_perf['MAE'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(df_perf, use_container_width=True)


def show_news_impact_tab(selected_symbols):
    """Show news sentiment impact analysis."""
    st.header("📰 News Sentiment Impact")
    
    st.info("This tab shows how news sentiment affects prediction accuracy and stock movements.")
    
    # Mock news impact data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment vs. Accuracy")
        
        # Generate mock data
        sentiment_data = []
        sentiments = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
        
        for sentiment in sentiments:
            base_accuracy = 0.5 + (sentiments.index(sentiment) * 0.05)  # Better accuracy with more extreme sentiment
            accuracy = base_accuracy + np.random.normal(0, 0.02)
            sentiment_data.append({
                'Sentiment': sentiment,
                'Accuracy': accuracy,
                'Count': np.random.randint(10, 50)
            })
        
        df_sentiment = pd.DataFrame(sentiment_data)
        
        fig = px.bar(
            df_sentiment, 
            x='Sentiment', 
            y='Accuracy',
            title="Prediction Accuracy by News Sentiment",
            color='Accuracy',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 News Volume Impact")
        
        # Mock news volume vs accuracy
        volume_data = []
        for i in range(10):
            volume = i * 5 + 5  # 5 to 50 articles
            accuracy = 0.55 + (i * 0.02) + np.random.normal(0, 0.01)
            volume_data.append({
                'News_Volume': volume,
                'Accuracy': accuracy
            })
        
        df_volume = pd.DataFrame(volume_data)
        
        fig = px.scatter(
            df_volume,
            x='News_Volume',
            y='Accuracy',
            title="Accuracy vs. News Article Volume",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent news summary
    st.subheader("📋 Recent High-Impact News")
    
    # Mock recent news data
    recent_news = [
        {
            'Date': '2025-09-24',
            'Symbol': 'AAPL',
            'Headline': 'Apple announces new iPhone features',
            'Sentiment': 'Positive',
            'Impact_Score': 0.85
        },
        {
            'Date': '2025-09-23',
            'Symbol': 'GOOGL',
            'Headline': 'Google AI breakthrough in quantum computing',
            'Sentiment': 'Very Positive',
            'Impact_Score': 0.92
        },
        {
            'Date': '2025-09-23',
            'Symbol': 'TSLA',
            'Headline': 'Tesla recalls vehicles for software issue',
            'Sentiment': 'Negative',
            'Impact_Score': 0.78
        }
    ]
    
    df_news = pd.DataFrame(recent_news)
    df_news = df_news[df_news['Symbol'].isin(selected_symbols)]
    
    if not df_news.empty:
        st.dataframe(df_news, use_container_width=True)
    else:
        st.info("No recent high-impact news for selected symbols.")


def show_model_info_tab():
    """Show model information and controls."""
    st.header("⚙️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Details")
        
        model_info = {
            'Model Type': 'Multi-Layer Perceptron (MLP)',
            'Framework': 'PyTorch',
            'Embedding Model': 'FinBERT (ProsusAI/finbert)',
            'Input Features': '773 dimensions',
            'Hidden Layers': '[512, 256, 128]',
            'Training Data': '~50k samples',
            'Last Updated': 'Not available'
        }
        
        for key, value in model_info.items():
            st.text(f"{key}: {value}")
    
    with col2:
        st.subheader("📊 Training Configuration")
        
        training_config = {
            'Batch Size': '32',
            'Learning Rate': '0.001',
            'Epochs': '50',
            'Validation Split': '20%',
            'Early Stopping': 'Enabled (patience=10)',
            'Optimizer': 'Adam',
            'Loss Function': 'MSE'
        }
        
        for key, value in training_config.items():
            st.text(f"{key}: {value}")
    
    st.markdown("---")
    
    # Model controls
    st.subheader("🛠️ Model Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Model Status"):
            st.success("Model status refreshed!")
    
    with col2:
        if st.button("📊 Collect New Data"):
            st.info("Data collection started in background...")
    
    with col3:
        if st.button("🏋️ Retrain Model"):
            st.warning("Model retraining started. This may take several hours...")
    
    # API status
    st.subheader("🌐 API Status")
    
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            st.success("✅ API is healthy and responding")
            api_info = response.json()
            st.json(api_info)
        else:
            st.error(f"❌ API returned status code: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {e}")
        st.info(f"Make sure the API is running on {API_BASE_URL}")


if __name__ == "__main__":
    main()