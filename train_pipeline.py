#!/usr/bin/env python3
"""
Complete training pipeline for stock prediction system.
This script orchestrates the entire process from data collection to model training.
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.data_collection.orchestrator import DataCollectionOrchestrator
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.training.stock_predictor import StockPredictor
from src.utils.config import config
from src.utils.logging_config import get_logger


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Stock Prediction Training Pipeline')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'], 
                        help='Stock symbols to train on')
    parser.add_argument('--start-date', type=str, default='2024-09-01', 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-08-31', 
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--sample-only', action='store_true', 
                        help='Run with sample data only (1 month)')
    parser.add_argument('--skip-data-collection', action='store_true', 
                        help='Skip data collection and use existing data')
    parser.add_argument('--skip-training', action='store_true', 
                        help='Skip model training (only collect and process data)')
    
    args = parser.parse_args()
    
    logger = get_logger(__name__)
    logger.info("[START] Starting Stock Prediction Training Pipeline")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    try:
        # Parse dates
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        # Adjust for sample mode
        if args.sample_only:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            logger.info(f"Sample mode: Using last 30 days ({start_date.date()} to {end_date.date()})")
        
        # Step 1: Data Collection
        if not args.skip_data_collection:
            logger.info("[STEP 1] Data Collection")
            
            orchestrator = DataCollectionOrchestrator()
            data_dir = project_root / "data" / "raw"
            
            if args.sample_only:
                summary = orchestrator.collect_sample_data(
                    symbols=args.symbols,
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=data_dir / "sample"
                )
                data_dir = data_dir / "sample"
            else:
                summary = orchestrator.collect_full_dataset(
                    symbols=args.symbols,
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=data_dir / "full",
                    parallel_workers=3
                )
                data_dir = data_dir / "full"
            
            logger.info(f"✅ Data collection completed: {summary}")
            
            if summary['symbols_processed'] == 0:
                logger.error("❌ No data collected. Exiting.")
                return False
        else:
            logger.info("[SKIP] Skipping data collection")
            data_dir = project_root / "data" / "raw" / ("sample" if args.sample_only else "full")
            
            if not data_dir.exists():
                logger.error(f"❌ Data directory not found: {data_dir}")
                return False
        
        # Step 2: Feature Engineering
        logger.info("[STEP 2] Feature Engineering")
        
        feature_engineer = FeatureEngineer()
        
        # Process news embeddings
        logger.info("Processing news embeddings...")
        news_embeddings = feature_engineer.news_embedder.process_news_for_symbols(data_dir)
        logger.info(f"Created embeddings for {len(news_embeddings)} symbols")
        
        # Create training dataset
        logger.info("Creating training dataset...")
        training_dataset = feature_engineer.create_training_dataset(data_dir)
        
        if training_dataset.empty:
            logger.error("❌ No training dataset created. Exiting.")
            return False
        
        # Save dataset
        processed_dir = project_root / "data" / "processed"
        dataset_path = processed_dir / f"training_dataset_{'sample' if args.sample_only else 'full'}.parquet"
        feature_engineer.save_training_dataset(training_dataset, dataset_path)
        
        logger.info(f"[SUCCESS] Feature engineering completed: {len(training_dataset)} samples")
        
        # Step 3: Model Training
        if not args.skip_training:
            logger.info("[STEP 3] Model Training")
            
            predictor = StockPredictor()
            
            # Prepare training data
            logger.info("Preparing training data...")
            features, targets = predictor.prepare_training_data(dataset_path=dataset_path)
            
            if len(features) == 0:
                logger.error("[ERROR] No training features created. Exiting.")
                return False
            
            logger.info(f"Training data prepared: {len(features)} samples, {features.shape[1]} features")
            
            # Train model
            logger.info("Training model...")
            history = predictor.train(features, targets)
            
            # Save model
            model_dir = project_root / "models" / "stock_predictor"
            predictor.save_model(model_dir)
            
            logger.info("[SUCCESS] Model training completed and saved")
            
            # Print training summary
            final_metrics = {
                'Final Train Loss': history['train_loss'][-1],
                'Final Val Loss': history['val_loss'][-1],
                'Final Val MAE': history['val_mae'][-1],
                'Final Val R²': history['val_r2'][-1]
            }
            
            logger.info("[METRICS] Final Training Metrics:")
            for metric, value in final_metrics.items():
                logger.info(f"  {metric}: {value:.6f}")
        
        else:
            logger.info("⏭️ Skipping model training")
        
        logger.info("[COMPLETE] Pipeline completed successfully!")
        
        # Next steps
        logger.info("\n[NEXT STEPS] Next Steps:")
        logger.info("1. Start the API server: uv run python -m src.api.main")
        logger.info("2. Start the dashboard: uv run streamlit run src/dashboard/app.py")
        logger.info("3. Test predictions via API or dashboard")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)