"""
FastAPI application for stock prediction service.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

from ..training.stock_predictor import StockPredictor
from ..data_collection.orchestrator import DataCollectionOrchestrator
from ..feature_engineering.feature_engineer import FeatureEngineer
from ..utils.config import config
from ..utils.logging_config import get_logger

# Initialize FastAPI app
app = FastAPI(
    title="Stock Prediction API",
    description="API for predicting stock prices using news sentiment and technical indicators",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('api.cors_origins', ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
predictor = None
logger = get_logger(__name__)


# Pydantic models for API
class StockSymbol(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    

class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    days_ahead: int = Field(7, description="Number of days to predict ahead", ge=1, le=30)
    include_news: bool = Field(True, description="Whether to include news sentiment")


class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    predicted_change_percent: float
    confidence_score: float
    prediction_date: str
    target_date: str
    features_used: Dict


class TrainingRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of stock symbols to train on")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    retrain: bool = Field(False, description="Whether to retrain from scratch")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global predictor
    logger.info("Starting Stock Prediction API...")
    
    try:
        # Try to load existing model
        model_path = Path("models/stock_predictor")
        if model_path.exists():
            predictor = StockPredictor()
            predictor.load_model(model_path)
            logger.info("Loaded existing model")
        else:
            logger.info("No existing model found. Model will need to be trained.")
            predictor = StockPredictor()
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        predictor = StockPredictor()


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None and predictor.model is not None,
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_stock_price(request: PredictionRequest):
    """Predict stock price for a given symbol."""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        logger.info(f"Predicting stock price for {request.symbol}")
        
        # Get current stock data
        from ..data_collection.stock_collector import StockCollector
        stock_collector = StockCollector()
        
        # Get recent data for features
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Get 30 days of data for technical indicators
        
        stock_data = stock_collector.fetch_stock_data(
            request.symbol, start_date, end_date
        )
        
        if stock_data is None or stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No stock data found for {request.symbol}")
        
        # Get the latest data point
        latest_data = stock_data.iloc[-1]
        current_price = latest_data['Close']
        
        # Prepare features for prediction
        feature_columns = predictor.feature_columns
        financial_features = latest_data[feature_columns].values
        
        # Add news features (placeholder - would need recent news data)
        news_features = np.zeros(768)  # FinBERT embedding size
        
        # Combine features
        combined_features = np.concatenate([financial_features, news_features]).reshape(1, -1)
        
        # Make prediction
        predicted_change = predictor.predict(combined_features)[0]
        predicted_price = current_price * (1 + predicted_change)
        
        # Calculate confidence (simplified)
        confidence_score = min(0.95, max(0.3, 0.8 - abs(predicted_change) * 5))
        
        # Target date
        target_date = (datetime.now() + timedelta(days=request.days_ahead)).strftime('%Y-%m-%d')
        
        return PredictionResponse(
            symbol=request.symbol,
            current_price=float(current_price),
            predicted_price=float(predicted_price),
            predicted_change_percent=float(predicted_change * 100),
            confidence_score=float(confidence_score),
            prediction_date=datetime.now().strftime('%Y-%m-%d'),
            target_date=target_date,
            features_used={
                "financial_indicators": len(financial_features),
                "news_embeddings": len(news_features),
                "include_news": request.include_news
            }
        )
        
    except Exception as e:
        logger.error(f"Error predicting stock price: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train the stock prediction model."""
    try:
        logger.info(f"Starting model training for symbols: {request.symbols}")
        
        # Parse dates
        start_date = datetime.strptime(request.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(request.end_date, '%Y-%m-%d')
        
        # Add training task to background
        background_tasks.add_task(
            _train_model_background, 
            request.symbols, 
            start_date, 
            end_date, 
            request.retrain
        )
        
        return {
            "message": "Training started in background",
            "symbols": request.symbols,
            "date_range": f"{request.start_date} to {request.end_date}",
            "status": "training"
        }
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed to start: {str(e)}")


async def _train_model_background(
    symbols: List[str], 
    start_date: datetime, 
    end_date: datetime, 
    retrain: bool
):
    """Background task for model training."""
    global predictor
    
    try:
        logger.info("Starting background model training...")
        
        # Step 1: Collect data
        orchestrator = DataCollectionOrchestrator()
        data_dir = Path("data/training")
        
        summary = orchestrator.collect_full_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            output_dir=data_dir
        )
        
        logger.info(f"Data collection complete: {summary}")
        
        # Step 2: Feature engineering
        feature_engineer = FeatureEngineer()
        training_dataset = feature_engineer.create_training_dataset(data_dir)
        
        if training_dataset.empty:
            logger.error("No training data created")
            return
        
        # Save dataset
        dataset_path = Path("data/processed/training_dataset.parquet")
        feature_engineer.save_training_dataset(training_dataset, dataset_path)
        
        # Step 3: Prepare training data
        features, targets = predictor.prepare_training_data(data_dir)
        
        # Step 4: Train model
        history = predictor.train(features, targets)
        
        # Step 5: Save model
        model_path = Path("models/stock_predictor")
        predictor.save_model(model_path)
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Background training failed: {e}")


@app.get("/model/status")
async def get_model_status():
    """Get current model status."""
    if predictor is None:
        return {"status": "not_initialized"}
    
    if predictor.model is None:
        return {"status": "not_trained"}
    
    return {
        "status": "ready",
        "feature_columns": predictor.feature_columns,
        "model_type": "MLP",
        "last_updated": "unknown"  # Would need to store this info
    }


@app.get("/symbols/supported")
async def get_supported_symbols():
    """Get list of supported stock symbols."""
    # This would ideally be dynamic based on available data
    symbols = config.get('data.stock_symbols', 'AAPL,GOOGL,MSFT,TSLA,AMZN').split(',')
    return {
        "symbols": symbols,
        "total": len(symbols)
    }


@app.post("/data/collect")
async def collect_data(symbols: List[str], background_tasks: BackgroundTasks):
    """Collect fresh data for specified symbols."""
    try:
        logger.info(f"Starting data collection for: {symbols}")
        
        # Add data collection task to background
        background_tasks.add_task(_collect_data_background, symbols)
        
        return {
            "message": "Data collection started in background",
            "symbols": symbols,
            "status": "collecting"
        }
        
    except Exception as e:
        logger.error(f"Error starting data collection: {e}")
        raise HTTPException(status_code=500, detail=f"Data collection failed to start: {str(e)}")


async def _collect_data_background(symbols: List[str]):
    """Background task for data collection."""
    try:
        orchestrator = DataCollectionOrchestrator()
        
        # Collect recent data (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        summary = orchestrator.collect_sample_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            output_dir=Path("data/raw/recent")
        )
        
        logger.info(f"Recent data collection complete: {summary}")
        
    except Exception as e:
        logger.error(f"Background data collection failed: {e}")


if __name__ == "__main__":
    import uvicorn
    
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 8000)
    debug = config.get('api.debug', False)
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )