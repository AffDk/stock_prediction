"""
Configuration management utilities for the stock prediction system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class Config:
    """Configuration manager that loads and manages configuration from YAML files and environment variables."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Path to configuration directory. Defaults to project_root/config
        """
        if config_dir is None:
            # Get project root directory (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"
        
        self.config_dir = Path(config_dir)
        self.project_root = self.config_dir.parent
        
        # Load environment variables
        env_file = self.project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        
        # Load configuration files
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files."""
        config = {}
        
        # Load main config
        main_config_file = self.config_dir / "config.yaml"
        if main_config_file.exists():
            with open(main_config_file, 'r') as f:
                config.update(yaml.safe_load(f))
        
        # Load stocks config
        stocks_config_file = self.config_dir / "stocks.yaml"
        if stocks_config_file.exists():
            with open(stocks_config_file, 'r') as f:
                stocks_config = yaml.safe_load(f)
                config['stocks'] = stocks_config
        
        # Substitute environment variables
        config = self._substitute_env_vars(config)
        
        return config
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables in configuration."""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        else:
            return obj
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'data.start_date')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_stock_symbols(self) -> list:
        """Get list of all stock symbols from configuration."""
        symbols = []
        
        # Get from environment variable first
        env_symbols = os.getenv('STOCK_SYMBOLS')
        if env_symbols:
            return [s.strip() for s in env_symbols.split(',')]
        
        # Get from stocks configuration
        stocks_config = self.get('stocks', {})
        for category in stocks_config.values():
            if isinstance(category, list):
                for stock in category:
                    if isinstance(stock, dict) and 'symbol' in stock:
                        symbols.append(stock['symbol'])
        
        return symbols
    
    def get_stock_config(self, symbol: str) -> Dict[str, Any]:
        """
        Get configuration for a specific stock symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Stock-specific configuration
        """
        # Get default settings
        default_config = self.get('stocks.default_settings', {})
        
        # Get stock-specific settings
        stock_specific = self.get(f'stocks.stock_configs.{symbol}', {})
        
        # Find stock details in categories
        stock_details = {}
        stocks_config = self.get('stocks', {})
        for category_name, category_stocks in stocks_config.items():
            if isinstance(category_stocks, list):
                for stock in category_stocks:
                    if isinstance(stock, dict) and stock.get('symbol') == symbol:
                        stock_details = stock
                        break
        
        # Merge configurations
        config = {**default_config, **stock_details, **stock_specific}
        return config
    
    @property
    def news_api_key(self) -> str:
        """Get NewsAPI key."""
        # Try environment variable first
        key = os.getenv('NEWS_API_KEY')
        if key:
            return key
        
        # Try to read from api_key_news file
        api_key_file = self.project_root / "api_key_news"
        if api_key_file.exists():
            return api_key_file.read_text().strip()
        
        raise ValueError("NEWS_API_KEY not found in environment or api_key_news file")
    
    @property
    def gcp_project_id(self) -> str:
        """Get GCP project ID."""
        project_id = os.getenv('GCP_PROJECT_ID')
        if not project_id:
            raise ValueError("GCP_PROJECT_ID not set in environment variables")
        return project_id
    
    @property
    def gcp_bucket_name(self) -> str:
        """Get GCS bucket name."""
        bucket_name = os.getenv('GCP_BUCKET_NAME')
        if not bucket_name:
            # Generate default bucket name
            bucket_name = f"{self.gcp_project_id}-stock-prediction"
        return bucket_name
    
    def get_data_path(self, data_type: str) -> Path:
        """
        Get path for different types of data.
        
        Args:
            data_type: Type of data ('raw', 'processed', 'training', 'models')
            
        Returns:
            Path to data directory
        """
        base_path = self.project_root / "data"
        
        if data_type == "raw":
            return base_path / self.get('data.raw_data_path', 'raw')
        elif data_type == "processed":
            return base_path / self.get('data.processed_data_path', 'processed')
        elif data_type == "training":
            return base_path / self.get('data.training_data_path', 'training')
        elif data_type == "models":
            return self.project_root / "models"
        else:
            return base_path / data_type

# Global configuration instance
config = Config()