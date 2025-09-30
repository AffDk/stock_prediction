"""
News data collector using NewsAPI.
"""

import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

from ..utils.config import config
from ..utils.logging_config import get_logger


class NewsCollector:
    """Collects news data from NewsAPI."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.api_key = self._get_api_key()
        self.base_url = "https://newsapi.org/v2"
        self.rate_limit_delay = 60 / config.get('news_api.rate_limit_requests_per_minute', 50)
        
    def _get_api_key(self) -> str:
        """Get NewsAPI key from various sources."""
        import os
        from pathlib import Path
        
        # Try environment variable first
        api_key = os.getenv('NEWS_API_KEY')
        if api_key and api_key != 'your_newsapi_key_here':
            return api_key
            
        # Try api_key_news file
        api_key_file = Path(__file__).parent.parent.parent / 'api_key_news'
        if api_key_file.exists():
            key = api_key_file.read_text().strip()
            if key and key != 'your_newsapi_key_here':
                return key
                
        raise ValueError("NewsAPI key not found. Please set NEWS_API_KEY environment variable or create api_key_news file.")
    
    def fetch_news_for_symbol(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict]:
        """
        Fetch news articles for a specific stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for news search
            end_date: End date for news search
            
        Returns:
            List of news articles
        """
        self.logger.info(f"Fetching news for {symbol} from {start_date.date()} to {end_date.date()}")
        
        articles = []
        current_date = start_date
        
        # NewsAPI has a limit on date range, so we fetch day by day for better coverage
        while current_date <= end_date:
            try:
                daily_articles = self._fetch_daily_news(symbol, current_date)
                articles.extend(daily_articles)
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                current_date += timedelta(days=1)
                
            except Exception as e:
                self.logger.error(f"Error fetching news for {symbol} on {current_date.date()}: {e}")
                current_date += timedelta(days=1)
                continue
        
        self.logger.info(f"Collected {len(articles)} articles for {symbol}")
        return articles
    
    def _fetch_daily_news(self, symbol: str, date: datetime) -> List[Dict]:
        """Fetch news for a specific symbol on a specific date."""
        
        # Create search queries for the symbol
        queries = [
            f'"{symbol}"',
            f'{symbol} stock',
            f'{self._get_company_name(symbol)} {symbol}'
        ]
        
        all_articles = []
        
        for query in queries:
            try:
                articles = self._make_api_request(query, date)
                # Filter articles that actually mention the symbol
                filtered_articles = self._filter_relevant_articles(articles, symbol)
                all_articles.extend(filtered_articles)
                
                # Small delay between queries
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.warning(f"Error with query '{query}' for {symbol}: {e}")
                continue
        
        # Remove duplicates based on URL
        unique_articles = []
        seen_urls = set()
        
        for article in all_articles:
            url = article.get('url', '')
            if url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        return unique_articles
    
    def _make_api_request(self, query: str, date: datetime) -> List[Dict]:
        """Make API request to NewsAPI."""
        
        params = {
            'q': query,
            'from': date.strftime('%Y-%m-%d'),
            'to': date.strftime('%Y-%m-%d'),
            'language': config.get('news_api.language', 'en'),
            'sortBy': config.get('news_api.sort_by', 'publishedAt'),
            'apiKey': self.api_key,
            'pageSize': 100  # Maximum allowed
        }
        
        # Add domain filters if configured
        domains = config.get('news_api.domains', [])
        if domains:
            params['domains'] = ','.join(domains)
        
        response = requests.get(f"{self.base_url}/everything", params=params)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('articles', [])
        elif response.status_code == 429:  # Rate limit
            self.logger.warning("Rate limit hit, waiting...")
            time.sleep(60)  # Wait 1 minute
            return self._make_api_request(query, date)  # Retry
        else:
            self.logger.error(f"API request failed: {response.status_code} - {response.text}")
            return []
    
    def _filter_relevant_articles(self, articles: List[Dict], symbol: str) -> List[Dict]:
        """Filter articles that are actually relevant to the symbol."""
        relevant_articles = []
        
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = article.get('content', '').lower()
            
            # Check if symbol appears in the text
            text_to_search = f"{title} {description} {content}"
            
            if (symbol.lower() in text_to_search or 
                self._get_company_name(symbol).lower() in text_to_search):
                
                # Add metadata
                article['symbol'] = symbol
                article['collection_date'] = datetime.now().isoformat()
                article['relevance_score'] = self._calculate_relevance_score(text_to_search, symbol)
                
                relevant_articles.append(article)
        
        return relevant_articles
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name for symbol (simplified mapping)."""
        company_names = {
            'AAPL': 'Apple',
            'GOOGL': 'Google',
            'MSFT': 'Microsoft',
            'TSLA': 'Tesla',
            'AMZN': 'Amazon',
            'NVDA': 'NVIDIA',
            'META': 'Meta',
            'NFLX': 'Netflix'
        }
        return company_names.get(symbol, symbol)
    
    def _calculate_relevance_score(self, text: str, symbol: str) -> float:
        """Calculate relevance score for an article."""
        score = 0.0
        
        # Count mentions of symbol
        score += text.lower().count(symbol.lower()) * 0.3
        
        # Count mentions of company name
        company_name = self._get_company_name(symbol)
        score += text.lower().count(company_name.lower()) * 0.2
        
        # Financial keywords boost
        financial_keywords = ['stock', 'price', 'earnings', 'revenue', 'profit', 'shares', 'market', 'trading']
        for keyword in financial_keywords:
            if keyword in text.lower():
                score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0