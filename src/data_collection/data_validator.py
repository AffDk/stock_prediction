"""
Data validator for stock prediction system.
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

from ..utils.logging_config import get_logger


class DataValidator:
    """Validates collected data quality and completeness."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate_symbol_data(
        self, 
        symbol: str, 
        stock_data: pd.DataFrame, 
        news_data: List[Dict]
    ) -> bool:
        """
        Validate data for a single symbol.
        
        Args:
            symbol: Stock symbol
            stock_data: Stock price data
            news_data: News articles data
            
        Returns:
            True if data is valid, False otherwise
        """
        is_valid = True
        
        # Validate stock data
        if not self._validate_stock_data(symbol, stock_data):
            is_valid = False
        
        # Validate news data
        if not self._validate_news_data(symbol, news_data):
            is_valid = False
        
        if is_valid:
            self.logger.info(f"✅ Data validation passed for {symbol}")
        else:
            self.logger.warning(f"❌ Data validation failed for {symbol}")
        
        return is_valid
    
    def _validate_stock_data(self, symbol: str, stock_data: pd.DataFrame) -> bool:
        """Validate stock data."""
        if stock_data is None or stock_data.empty:
            self.logger.error(f"No stock data for {symbol}")
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in stock_data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing stock data columns for {symbol}: {missing_columns}")
            return False
        
        # Check for null values in critical columns
        critical_nulls = stock_data[['Close', 'Volume']].isnull().sum()
        if critical_nulls.any():
            self.logger.warning(f"Null values in critical columns for {symbol}: {critical_nulls.to_dict()}")
        
        # Check for reasonable price ranges
        if (stock_data['Close'] <= 0).any():
            self.logger.error(f"Invalid close prices (<=0) found for {symbol}")
            return False
        
        # Check for reasonable volume
        if (stock_data['Volume'] < 0).any():
            self.logger.error(f"Invalid volume (<0) found for {symbol}")
            return False
        
        self.logger.debug(f"Stock data validation passed for {symbol}: {len(stock_data)} records")
        return True
    
    def _validate_news_data(self, symbol: str, news_data: List[Dict]) -> bool:
        """Validate news data."""
        if not news_data:
            self.logger.warning(f"No news data for {symbol}")
            return True  # Not having news is not a hard failure
        
        valid_articles = 0
        required_fields = ['title', 'publishedAt', 'url']
        
        for i, article in enumerate(news_data):
            if not isinstance(article, dict):
                self.logger.warning(f"Article {i} for {symbol} is not a dictionary")
                continue
            
            missing_fields = [field for field in required_fields if field not in article or not article[field]]
            if missing_fields:
                self.logger.warning(f"Article {i} for {symbol} missing fields: {missing_fields}")
                continue
            
            # Validate date format
            try:
                datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                self.logger.warning(f"Invalid date format in article {i} for {symbol}")
                continue
            
            valid_articles += 1
        
        validation_rate = valid_articles / len(news_data) if news_data else 0
        if validation_rate < 0.8:  # At least 80% of articles should be valid
            self.logger.warning(f"Low news validation rate for {symbol}: {validation_rate:.2%}")
        
        self.logger.debug(f"News validation for {symbol}: {valid_articles}/{len(news_data)} valid articles")
        return True
    
    def validate_dataset_completeness(
        self, 
        symbols: List[str], 
        data_dir: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict:
        """
        Validate completeness of the entire dataset.
        
        Args:
            symbols: List of symbols to check
            data_dir: Directory containing the data
            start_date: Expected start date
            end_date: Expected end date
            
        Returns:
            Validation summary dictionary
        """
        from pathlib import Path
        
        summary = {
            'total_symbols': len(symbols),
            'symbols_with_data': 0,
            'symbols_missing': [],
            'date_coverage': {},
            'data_quality_issues': []
        }
        
        data_path = Path(data_dir)
        
        for symbol in symbols:
            symbol_dir = data_path / symbol
            stock_file = symbol_dir / f"{symbol}_stock_data.parquet"
            news_file = symbol_dir / f"{symbol}_news_data.parquet"
            
            if not symbol_dir.exists():
                summary['symbols_missing'].append(symbol)
                continue
            
            if stock_file.exists():
                try:
                    stock_data = pd.read_parquet(stock_file)
                    
                    # Check date coverage
                    if 'Date' in stock_data.columns:
                        min_date = pd.to_datetime(stock_data['Date']).min()
                        max_date = pd.to_datetime(stock_data['Date']).max()
                        summary['date_coverage'][symbol] = {
                            'start': min_date.strftime('%Y-%m-%d'),
                            'end': max_date.strftime('%Y-%m-%d'),
                            'days': len(stock_data)
                        }
                    
                    summary['symbols_with_data'] += 1
                    
                except Exception as e:
                    summary['data_quality_issues'].append(f"{symbol}: {str(e)}")
            else:
                summary['symbols_missing'].append(symbol)
        
        # Calculate overall completeness
        summary['completeness_rate'] = summary['symbols_with_data'] / summary['total_symbols']
        
        self.logger.info(f"Dataset validation complete: {summary['completeness_rate']:.2%} completeness")
        return summary