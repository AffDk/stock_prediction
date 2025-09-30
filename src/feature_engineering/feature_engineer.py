"""
Feature engineering module for stock prediction.
Handles news embedding and feature combination.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

from transformers import AutoTokenizer, AutoModel
import torch

from ..utils.logging_config import get_logger
from ..utils.config import config


class NewsEmbedder:
    """Handles news text embedding using FinBERT."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model_name = config.get('model.news_embedding.model_name', 'ProsusAI/finbert')
        self.max_length = config.get('model.news_embedding.max_length', 512)
        self.batch_size = config.get('model.news_embedding.batch_size_embedding', 16)
        
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load FinBERT model and tokenizer."""
        try:
            self.logger.info(f"Loading FinBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            self.logger.info("FinBERT model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading FinBERT model: {e}")
            raise
    
    def embed_news_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed news texts using FinBERT.
        
        Args:
            texts: List of news texts to embed
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts."""
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings.tolist()
    
    def process_news_for_symbols(self, data_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Process news data for all symbols and create embeddings.
        
        Args:
            data_dir: Directory containing raw news data
            
        Returns:
            Dictionary mapping symbol to news embeddings DataFrame
        """
        self.logger.info("Processing news data for all symbols...")
        
        symbol_embeddings = {}
        
        for symbol_dir in data_dir.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                news_file = symbol_dir / f"{symbol}_news_data.parquet"
                
                if news_file.exists():
                    try:
                        # Load news data
                        news_df = pd.read_parquet(news_file)
                        
                        if not news_df.empty:
                            # Create embeddings
                            embeddings_df = self._process_symbol_news(news_df, symbol)
                            symbol_embeddings[symbol] = embeddings_df
                            
                            # Save embeddings
                            embeddings_file = symbol_dir / f"{symbol}_news_embeddings.parquet"
                            embeddings_df.to_parquet(embeddings_file)
                            
                            self.logger.info(f"Processed {len(embeddings_df)} news embeddings for {symbol}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing news for {symbol}: {e}")
        
        return symbol_embeddings
    
    def _process_symbol_news(self, news_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process news data for a single symbol."""
        
        # Combine title and description for embedding
        texts = []
        for _, row in news_df.iterrows():
            title = row.get('title', '')
            description = row.get('description', '')
            text = f"{title}. {description}".strip()
            texts.append(text)
        
        # Create embeddings
        embeddings = self.embed_news_texts(texts)
        
        # Create DataFrame with embeddings
        embedding_columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]
        embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns)
        
        # Add metadata
        embeddings_df['symbol'] = symbol
        embeddings_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
        embeddings_df['url'] = news_df['url']
        embeddings_df['relevance_score'] = news_df.get('relevance_score', 1.0)
        
        return embeddings_df


class FeatureEngineer:
    """Combines stock data with news embeddings to create training features."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.news_embedder = NewsEmbedder()
    
    def create_training_dataset(
        self, 
        stock_data_dir: Path, 
        news_data_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Create complete training dataset combining stock and news features.
        
        Args:
            stock_data_dir: Directory with stock data
            news_data_dir: Directory with news data (if different from stock_data_dir)
            
        Returns:
            Combined training dataset
        """
        if news_data_dir is None:
            news_data_dir = stock_data_dir
        
        self.logger.info("Creating training dataset...")
        
        # Process news embeddings if not already done
        if not self._check_embeddings_exist(news_data_dir):
            self.news_embedder.process_news_for_symbols(news_data_dir)
        
        # Combine all data
        combined_data = []
        
        for symbol_dir in stock_data_dir.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                
                # Load stock data
                stock_file = symbol_dir / f"{symbol}_stock_data.parquet"
                if not stock_file.exists():
                    continue
                
                stock_df = pd.read_parquet(stock_file)
                
                # Load news embeddings
                embeddings_file = symbol_dir / f"{symbol}_news_embeddings.parquet"
                if embeddings_file.exists():
                    embeddings_df = pd.read_parquet(embeddings_file)
                else:
                    embeddings_df = pd.DataFrame()
                
                # Combine for this symbol
                symbol_data = self._combine_symbol_data(stock_df, embeddings_df, symbol)
                if not symbol_data.empty:
                    combined_data.append(symbol_data)
        
        if combined_data:
            final_dataset = pd.concat(combined_data, ignore_index=True)
            self.logger.info(f"Created training dataset with {len(final_dataset)} samples")
            return final_dataset
        else:
            self.logger.warning("No training data created!")
            return pd.DataFrame()
    
    def _check_embeddings_exist(self, data_dir: Path) -> bool:
        """Check if news embeddings already exist."""
        for symbol_dir in data_dir.iterdir():
            if symbol_dir.is_dir():
                embeddings_file = symbol_dir / f"{symbol_dir.name}_news_embeddings.parquet"
                if not embeddings_file.exists():
                    return False
        return True
    
    def _combine_symbol_data(
        self, 
        stock_df: pd.DataFrame, 
        embeddings_df: pd.DataFrame, 
        symbol: str
    ) -> pd.DataFrame:
        """Combine stock and news data for a single symbol."""
        
        stock_df = stock_df.copy()
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df = stock_df.sort_values('Date').reset_index(drop=True)
        
        # Parameters
        prediction_horizon = config.get('data.prediction_horizon_days', 7)
        news_lookback = config.get('data.news_lookback_days', 7)
        
        combined_rows = []
        
        for i in range(len(stock_df) - prediction_horizon):
            current_date = stock_df.iloc[i]['Date']
            future_idx = i + prediction_horizon
            
            if future_idx >= len(stock_df):
                continue
            
            # Get current stock features
            current_stock = stock_df.iloc[i]
            
            # Get future price for target
            future_price = stock_df.iloc[future_idx]['Close']
            current_price = current_stock['Close']
            target = (future_price - current_price) / current_price  # Percentage change
            
            # Get news features for the lookback period
            news_start = current_date - timedelta(days=news_lookback)
            news_end = current_date
            
            news_features = self._get_news_features(
                embeddings_df, news_start, news_end
            )
            
            # Create combined row
            row = current_stock.to_dict()
            row.update(news_features)
            row['target'] = target
            row['prediction_date'] = current_date
            row['target_date'] = stock_df.iloc[future_idx]['Date']
            
            combined_rows.append(row)
        
        return pd.DataFrame(combined_rows)
    
    def _get_news_features(
        self, 
        embeddings_df: pd.DataFrame, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict:
        """Get aggregated news features for a date range."""
        
        if embeddings_df.empty:
            # Return zero embeddings if no news data
            embedding_dim = config.get('model.news_embedding.embedding_dim', 768)
            return {f'news_embedding_{i}': 0.0 for i in range(embedding_dim)}
        
        # Filter news by date
        embeddings_df['publishedAt'] = pd.to_datetime(embeddings_df['publishedAt'])
        mask = (embeddings_df['publishedAt'] >= start_date) & (embeddings_df['publishedAt'] <= end_date)
        relevant_news = embeddings_df[mask]
        
        if relevant_news.empty:
            # Return zero embeddings if no relevant news
            embedding_dim = config.get('model.news_embedding.embedding_dim', 768)
            return {f'news_embedding_{i}': 0.0 for i in range(embedding_dim)}
        
        # Get embedding columns
        embedding_cols = [col for col in relevant_news.columns if col.startswith('embedding_')]
        
        if not embedding_cols:
            embedding_dim = config.get('model.news_embedding.embedding_dim', 768)
            return {f'news_embedding_{i}': 0.0 for i in range(embedding_dim)}
        
        # Aggregate embeddings (weighted by relevance score)
        if 'relevance_score' in relevant_news.columns:
            weights = relevant_news['relevance_score'].values
            weighted_embeddings = relevant_news[embedding_cols].multiply(weights, axis=0)
            aggregated = weighted_embeddings.sum() / weights.sum()
        else:
            aggregated = relevant_news[embedding_cols].mean()
        
        # Convert to dictionary
        news_features = {}
        for i, col in enumerate(embedding_cols):
            news_features[f'news_embedding_{i}'] = aggregated[col]
        
        return news_features
    
    def save_training_dataset(self, dataset: pd.DataFrame, output_path: Path):
        """Save the training dataset."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(output_path)
        self.logger.info(f"Training dataset saved to {output_path}")
        
        # Save summary statistics
        summary = {
            'total_samples': len(dataset),
            'symbols': dataset['Symbol'].nunique() if 'Symbol' in dataset.columns else 0,
            'date_range': {
                'start': dataset['prediction_date'].min().isoformat() if 'prediction_date' in dataset.columns else None,
                'end': dataset['prediction_date'].max().isoformat() if 'prediction_date' in dataset.columns else None
            },
            'target_stats': {
                'mean': float(dataset['target'].mean()) if 'target' in dataset.columns else None,
                'std': float(dataset['target'].std()) if 'target' in dataset.columns else None,
                'min': float(dataset['target'].min()) if 'target' in dataset.columns else None,
                'max': float(dataset['target'].max()) if 'target' in dataset.columns else None
            },
            'created_at': datetime.now().isoformat()
        }
        
        import json
        summary_path = output_path.parent / f"{output_path.stem}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Dataset summary saved to {summary_path}")