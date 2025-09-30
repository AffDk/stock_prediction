"""
Data Collection Orchestrator for Stock Prediction System
Handles fetching news and stock data with proper rate limiting and storage.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

from ..utils.config import config
from ..utils.logging_config import get_logger
from .news_collector import NewsCollector
from .stock_collector import StockCollector
from .data_validator import DataValidator


class DataCollectionOrchestrator:
    """Orchestrates the collection of news and stock data."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.news_collector = NewsCollector()
        self.stock_collector = StockCollector()
        self.validator = DataValidator()
        
    def collect_sample_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        output_dir: Path
    ) -> Dict:
        """
        Collect sample data for testing and validation.
        
        Args:
            symbols: List of stock symbols to collect data for
            start_date: Start date for data collection
            end_date: End date for data collection
            output_dir: Directory to save collected data
            
        Returns:
            Summary dictionary with collection statistics
        """
        self.logger.info(f"Starting sample data collection for {len(symbols)} symbols")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'symbols_processed': 0,
            'symbols_failed': 0,
            'total_news_articles': 0,
            'total_stock_records': 0,
            'failed_symbols': []
        }
        
        for symbol in symbols:
            try:
                self.logger.info(f"Processing symbol: {symbol}")
                symbol_data = self._collect_symbol_data(symbol, start_date, end_date)
                
                if symbol_data:
                    # Save data
                    self._save_symbol_data(symbol_data, symbol, output_dir)
                    
                    # Update summary
                    summary['symbols_processed'] += 1
                    summary['total_news_articles'] += len(symbol_data.get('news', []))
                    summary['total_stock_records'] += len(symbol_data.get('stock_data', []))
                    
                    self.logger.info(f"Successfully processed {symbol}")
                else:
                    summary['symbols_failed'] += 1
                    summary['failed_symbols'].append(symbol)
                    self.logger.warning(f"Failed to collect data for {symbol}")
                    
                # Rate limiting - be nice to APIs
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                summary['symbols_failed'] += 1
                summary['failed_symbols'].append(symbol)
        
        self.logger.info("Sample data collection completed")
        return summary
    
    def collect_full_dataset(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        output_dir: Path,
        parallel_workers: int = 3
    ) -> Dict:
        """
        Collect full dataset for training.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for collection
            end_date: End date for collection
            output_dir: Output directory
            parallel_workers: Number of parallel workers
            
        Returns:
            Collection summary
        """
        self.logger.info(f"Starting full dataset collection for {len(symbols)} symbols")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use ThreadPoolExecutor for parallel processing
        summary = {
            'symbols_processed': 0,
            'symbols_failed': 0,
            'total_news_articles': 0,
            'total_stock_records': 0,
            'failed_symbols': []
        }
        
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self._collect_symbol_data_with_retry, 
                    symbol, start_date, end_date
                ): symbol for symbol in symbols
            }
            
            # Process results as they complete
            for future in futures:
                symbol = futures[future]
                try:
                    symbol_data = future.result()
                    if symbol_data:
                        self._save_symbol_data(symbol_data, symbol, output_dir)
                        summary['symbols_processed'] += 1
                        summary['total_news_articles'] += len(symbol_data.get('news', []))
                        summary['total_stock_records'] += len(symbol_data.get('stock_data', []))
                        self.logger.info(f"✅ Completed {symbol}")
                    else:
                        summary['symbols_failed'] += 1
                        summary['failed_symbols'].append(symbol)
                        self.logger.warning(f"❌ Failed {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    summary['symbols_failed'] += 1
                    summary['failed_symbols'].append(symbol)
        
        return summary
    
    def _collect_symbol_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[Dict]:
        """Collect data for a single symbol."""
        try:
            # Collect stock data
            stock_data = self.stock_collector.fetch_stock_data(
                symbol, start_date, end_date
            )
            
            if stock_data is None or stock_data.empty:
                self.logger.warning(f"No stock data found for {symbol}")
                return None
            
            # Collect news data
            news_data = self.news_collector.fetch_news_for_symbol(
                symbol, start_date, end_date
            )
            
            # Validate data
            if not self.validator.validate_symbol_data(symbol, stock_data, news_data):
                self.logger.warning(f"Data validation failed for {symbol}")
                return None
            
            return {
                'symbol': symbol,
                'stock_data': stock_data,
                'news': news_data,
                'collection_date': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}: {e}")
            return None
    
    def _collect_symbol_data_with_retry(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        max_retries: int = 3
    ) -> Optional[Dict]:
        """Collect symbol data with retry logic."""
        for attempt in range(max_retries):
            try:
                result = self._collect_symbol_data(symbol, start_date, end_date)
                if result:
                    return result
                    
                # If no result, wait before retry
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.info(f"Retrying {symbol} in {wait_time} seconds...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        return None
    
    def _save_symbol_data(self, symbol_data: Dict, symbol: str, output_dir: Path):
        """Save collected data for a symbol."""
        symbol_dir = output_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        
        # Save stock data
        if not symbol_data['stock_data'].empty:
            stock_file = symbol_dir / f"{symbol}_stock_data.parquet"
            symbol_data['stock_data'].to_parquet(stock_file)
        
        # Save news data
        if symbol_data['news']:
            news_df = pd.DataFrame(symbol_data['news'])
            news_file = symbol_dir / f"{symbol}_news_data.parquet"
            news_df.to_parquet(news_file)
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'collection_date': symbol_data['collection_date'].isoformat(),
            'stock_records': len(symbol_data['stock_data']),
            'news_articles': len(symbol_data['news'])
        }
        
        import json
        metadata_file = symbol_dir / f"{symbol}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.debug(f"Saved data for {symbol} to {symbol_dir}")