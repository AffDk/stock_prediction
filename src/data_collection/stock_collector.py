"""
Stock data collector using yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import time

from ..utils.logging_config import get_logger


class StockCollector:
    """Collects stock data using yfinance."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
    def fetch_stock_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date
            end_date: End date
            interval: Data interval ('1d', '1h', etc.)
            
        Returns:
            DataFrame with stock data or None if failed
        """
        try:
            self.logger.info(f"Fetching stock data for {symbol}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval,
                auto_adjust=True,
                prepost=False
            )
            
            if data.empty:
                self.logger.warning(f"No stock data found for {symbol}")
                return None
            
            # Add symbol column
            data['Symbol'] = symbol
            
            # Calculate additional features
            data = self._add_technical_indicators(data)
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            self.logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch stock data for multiple symbols."""
        all_data = []
        
        for symbol in symbols:
            try:
                data = self.fetch_stock_data(symbol, start_date, end_date, interval)
                if data is not None:
                    all_data.append(data)
                    
                # Small delay to be nice to the API
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error fetching {symbol}: {e}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to stock data."""
        try:
            # Simple Moving Averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            
            # RSI (Relative Strength Index)
            data['RSI_14'] = self._calculate_rsi(data['Close'], 14)
            
            # MACD
            data = self._add_macd(data)
            
            # Bollinger Bands
            data = self._add_bollinger_bands(data)
            
            # Volume indicators
            data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
            
            # Volatility (20-day)
            data['Volatility_20'] = data['Close'].pct_change().rolling(window=20).std() * (252 ** 0.5)
            
            # Price returns
            data['Return_1d'] = data['Close'].pct_change()
            data['Return_5d'] = data['Close'].pct_change(periods=5)
            data['Return_20d'] = data['Close'].pct_change(periods=20)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _add_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicators."""
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        data['MACD'] = ema_12 - ema_26
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        return data
    
    def _add_bollinger_bands(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add Bollinger Bands."""
        sma = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        data['BB_Upper'] = sma + (std * 2)
        data['BB_Lower'] = sma - (std * 2)
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / data['BB_Width']
        return data
    
    def get_company_info(self, symbol: str) -> dict:
        """Get company information for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD')
            }
        except Exception as e:
            self.logger.error(f"Error getting company info for {symbol}: {e}")
            return {'symbol': symbol}