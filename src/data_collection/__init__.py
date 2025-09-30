"""
Data collection module for stock prediction system.
This module handles collecting news and stock data from various APIs.
"""

import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from newsapi import NewsApiClient
import yfinance as yf

from ..utils.config import config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class NewsDataCollector:
    """Collect news data from NewsAPI."""
    
    def __init__(self, api_key: str):
        """Initialize news data collector."""
        self.client = NewsApiClient(api_key=api_key)
        self.rate_limit_delay = 60 / config.get('news_api.rate_limit_requests_per_minute', 50)
        
    def collect_news_for_stock(
        self, 
        symbol: str, 
        keywords: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Collect news articles for a specific stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            keywords: List of keywords to search for
            start_date: Start date for news collection
            end_date: End date for news collection
            
        Returns:
            List of news articles
        """
        articles = []
        
        # Construct search query
        query_terms = [symbol] + keywords
        query = f"({' OR '.join(query_terms)})"
        
        try:
            logger.info(f"Collecting news for {symbol} from {start_date} to {end_date}")
            
            # API call with rate limiting
            response = self.client.get_everything(
                q=query,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt',
                page_size=100
            )
            
            if response['status'] == 'ok':
                articles.extend(response['articles'])
                logger.info(f"Collected {len(articles)} articles for {symbol}")
            else:
                logger.error(f"API error for {symbol}: {response}")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
        except Exception as e:
            logger.error(f"Error collecting news for {symbol}: {e}")
        
        return articles
    
    def filter_relevant_articles(
        self, 
        articles: List[Dict[str, Any]], 
        symbol: str,
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter articles for relevance to the stock.
        
        Args:
            articles: List of news articles
            symbol: Stock symbol
            keywords: List of relevant keywords
            
        Returns:
            Filtered list of articles
        """
        filtered_articles = []
        
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = article.get('content', '').lower()
            
            # Check if article mentions the symbol or keywords
            text_content = f"{title} {description} {content}"
            
            relevance_score = 0
            if symbol.lower() in text_content:
                relevance_score += 3
            
            for keyword in keywords:
                if keyword.lower() in text_content:
                    relevance_score += 1
            
            # Filter based on relevance threshold
            min_relevance = config.get('stocks.default_settings.news_search.relevance_threshold', 0.6)
            if relevance_score >= min_relevance:
                article['relevance_score'] = relevance_score
                article['symbol'] = symbol
                filtered_articles.append(article)
        
        logger.info(f"Filtered {len(filtered_articles)} relevant articles for {symbol}")
        return filtered_articles

class StockDataCollector:
    """Collect stock price and technical indicator data."""
    
    def __init__(self):
        """Initialize stock data collector."""
        pass
    
    def collect_stock_data(
        self, 
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Collect stock price data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data collection
            end_date: End date for data collection
            interval: Data interval ('1d', '1h', etc.)
            
        Returns:
            DataFrame with stock data or None if failed
        """
        try:
            logger.info(f"Collecting stock data for {symbol} from {start_date} to {end_date}")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval
            )
            
            if data.empty:
                logger.warning(f"No stock data found for {symbol}")
                return None
            
            # Add symbol column
            data['Symbol'] = symbol
            data = data.reset_index()
            
            logger.info(f"Collected {len(data)} rows of stock data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting stock data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for stock data.
        
        Args:
            data: DataFrame with stock price data
            
        Returns:
            DataFrame with technical indicators added
        """
        try:
            # Simple Moving Average (20 days)
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            
            # RSI (14 days)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            
            # Volume Moving Average
            data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
            
            # Volatility (20-day rolling standard deviation of returns)
            data['Returns'] = data['Close'].pct_change()
            data['Volatility_20'] = data['Returns'].rolling(window=20).std()
            
            logger.info(f"Calculated technical indicators for {data['Symbol'].iloc[0] if 'Symbol' in data.columns else 'unknown'}")
            return data
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return data

class DataCollectionOrchestrator:
    """Orchestrate the collection of both news and stock data."""
    
    def __init__(self):
        """Initialize data collection orchestrator."""
        self.news_collector = NewsDataCollector(config.news_api_key)
        self.stock_collector = StockDataCollector()
    
    def collect_sample_data(
        self, 
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Collect sample data for testing and prototyping.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data collection
            end_date: End date for data collection
            output_dir: Directory to save the collected data
            
        Returns:
            Summary of collected data
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'symbols_processed': [],
            'symbols_failed': [],
            'total_news_articles': 0,
            'total_stock_records': 0
        }
        
        all_news_data = []
        all_stock_data = []
        
        for symbol in symbols:
            try:
                logger.info(f"Processing symbol: {symbol}")
                
                # Get stock configuration
                stock_config = config.get_stock_config(symbol)
                keywords = stock_config.get('keywords', [])
                
                # Collect news data
                news_articles = self.news_collector.collect_news_for_stock(
                    symbol, keywords, start_date, end_date
                )
                
                # Filter relevant articles
                relevant_articles = self.news_collector.filter_relevant_articles(
                    news_articles, symbol, keywords
                )
                
                all_news_data.extend(relevant_articles)
                summary['total_news_articles'] += len(relevant_articles)
                
                # Collect stock data
                stock_data = self.stock_collector.collect_stock_data(
                    symbol, start_date, end_date
                )
                
                if stock_data is not None:
                    # Calculate technical indicators
                    stock_data = self.stock_collector.calculate_technical_indicators(stock_data)
                    all_stock_data.append(stock_data)
                    summary['total_stock_records'] += len(stock_data)
                
                summary['symbols_processed'].append(symbol)
                
            except Exception as e:
                logger.error(f"Failed to process symbol {symbol}: {e}")
                summary['symbols_failed'].append(symbol)
        
        # Save collected data
        if all_news_data:
            news_df = pd.DataFrame(all_news_data)
            news_file = output_dir / "sample_news_data.parquet"
            news_df.to_parquet(news_file, index=False)
            logger.info(f"Saved {len(news_df)} news articles to {news_file}")
        
        if all_stock_data:
            stock_df = pd.concat(all_stock_data, ignore_index=True)
            stock_file = output_dir / "sample_stock_data.parquet"
            stock_df.to_parquet(stock_file, index=False)
            logger.info(f"Saved {len(stock_df)} stock records to {stock_file}")
        
        return summary