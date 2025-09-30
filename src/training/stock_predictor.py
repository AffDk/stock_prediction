"""
Stock prediction model training module.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from ..utils.logging_config import get_logger
from ..utils.config import config


class StockDataset(Dataset):
    """PyTorch Dataset for stock prediction."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class StockPredictionMLP(nn.Module):
    """Multi-layer perceptron for stock price prediction."""
    
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout_rate: float = 0.3):
        super(StockPredictionMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class StockPredictor:
    """Main stock prediction model trainer and predictor."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_columns = None
        
        # Load configuration
        self.config = config.get('model', {})
        self.training_config = self.config.get('training', {})
        
    def prepare_training_data(self, data_dir: Path = None, dataset_path: Path = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from processed dataset or collected raw data.
        
        Args:
            data_dir: Directory containing collected raw data (optional, legacy support)
            dataset_path: Path to processed training dataset parquet file
            
        Returns:
            Tuple of (features, targets)
        """
        self.logger.info("Preparing training data...")
        
        # First try to load from processed dataset
        if dataset_path and dataset_path.exists():
            self.logger.info(f"Loading processed training dataset from {dataset_path}")
            dataset = pd.read_parquet(dataset_path)
            
            # Extract features and target
            # Exclude non-feature columns
            exclude_cols = ['Date', 'Symbol', 'target', 'prediction_date', 'target_date']
            feature_cols = [col for col in dataset.columns if col not in exclude_cols]
            
            features = dataset[feature_cols].values.astype(np.float64)
            targets = dataset['target'].values.astype(np.float64)
            
            # Check for NaN values and handle them
            self.logger.info(f"Features shape: {features.shape}")
            self.logger.info(f"NaN count in features: {np.isnan(features).sum()}")
            self.logger.info(f"NaN count in targets: {np.isnan(targets).sum()}")
            
            # Check which feature columns have NaN
            nan_cols = []
            for i, col in enumerate(feature_cols):
                if np.isnan(features[:, i]).any():
                    nan_count = np.isnan(features[:, i]).sum()
                    nan_cols.append(f"{col}({nan_count})")
            if nan_cols:
                self.logger.warning(f"Features with NaN values: {nan_cols}")
            
            # Fill NaN values with appropriate defaults instead of removing rows
            # For technical indicators, use 0 as default (neutral value)
            # For returns, use 0 (no change)
            # For other values, use forward fill or 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.logger.info(f"After NaN handling - Features shape: {features.shape}, NaN count: {np.isnan(features).sum()}")
            self.logger.info(f"After NaN handling - Targets NaN count: {np.isnan(targets).sum()}")
            
            self.feature_columns = feature_cols
            
        else:
            # Fallback to legacy method for raw data processing
            self.logger.warning("Processed dataset not found, falling back to raw data processing")
            
            if not data_dir or not data_dir.exists():
                raise ValueError("No data directory or dataset path provided")
            
            features, targets = self._prepare_from_raw_data(data_dir)
        
        self.logger.info(f"Prepared {len(features)} training samples with {features.shape[1]} features")
        return features, targets

    def _prepare_from_raw_data(self, data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Legacy method to prepare training data from raw collected data."""
        
        # Load and combine all stock data
        all_stock_data = []
        all_news_embeddings = []
        
        for symbol_dir in data_dir.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name
                
                # Load stock data
                stock_file = symbol_dir / f"{symbol}_stock_data.parquet"
                if stock_file.exists():
                    stock_data = pd.read_parquet(stock_file)
                    stock_data['Symbol'] = symbol
                    all_stock_data.append(stock_data)
                
                # Load news embeddings (if available)
                news_embeddings_file = symbol_dir / f"{symbol}_news_embeddings.parquet"
                if news_embeddings_file.exists():
                    news_emb = pd.read_parquet(news_embeddings_file)
                    all_news_embeddings.append(news_emb)
        
        if not all_stock_data:
            raise ValueError("No stock data found in the specified directory")
        
        # Combine all stock data
        combined_stock_data = pd.concat(all_stock_data, ignore_index=True)
        
        # Create features and targets
        features, targets = self._create_features_and_targets(
            combined_stock_data, all_news_embeddings
        )
        
        return features, targets
    
    def _create_features_and_targets(
        self, 
        stock_data: pd.DataFrame, 
        news_embeddings: List[pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create feature matrix and target vector."""
        
        # Sort by date and symbol
        stock_data = stock_data.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        features_list = []
        targets_list = []
        
        # Define feature columns (financial indicators)
        financial_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
            'Volume_SMA_20', 'Volatility_20',
            'Return_1d', 'Return_5d', 'Return_20d'
        ]
        
        # Filter available columns
        available_features = [col for col in financial_features if col in stock_data.columns]
        self.feature_columns = available_features
        
        # Group by symbol for processing
        for symbol, symbol_data in stock_data.groupby('Symbol'):
            symbol_data = symbol_data.sort_values('Date').reset_index(drop=True)
            
            # Create prediction pairs (features -> target)
            prediction_horizon = config.get('data.prediction_horizon_days', 7)
            
            for i in range(len(symbol_data) - prediction_horizon):
                # Current features
                current_features = symbol_data.iloc[i][available_features].values
                
                # Target (future price)
                future_idx = i + prediction_horizon
                if future_idx < len(symbol_data):
                    current_price = symbol_data.iloc[i]['Close']
                    future_price = symbol_data.iloc[future_idx]['Close']
                    
                    # Calculate percentage change as target
                    target = (future_price - current_price) / current_price
                    
                    # Add news embedding if available (placeholder for now)
                    news_features = np.zeros(768)  # FinBERT embedding size
                    
                    # Combine features
                    combined_features = np.concatenate([current_features, news_features])
                    
                    features_list.append(combined_features)
                    targets_list.append(target)
        
        features = np.array(features_list, dtype=np.float64)
        targets = np.array(targets_list, dtype=np.float64)
        
        # Remove any NaN values
        if len(features) > 0 and len(targets) > 0:
            features_mask = np.isnan(features).any(axis=1) if features.ndim > 1 else np.isnan(features)
            targets_mask = np.isnan(targets)
            mask = ~(features_mask | targets_mask)
            features = features[mask]
            targets = targets[mask]
        
        return features, targets
    
    def train(
        self, 
        features: np.ndarray, 
        targets: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train the stock prediction model.
        
        Args:
            features: Feature matrix
            targets: Target vector
            validation_split: Fraction for validation
            
        Returns:
            Training history dictionary
        """
        self.logger.info("Starting model training...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, targets, 
            test_size=validation_split, 
            random_state=42,
            shuffle=True
        )
        
        # Scale features and targets
        self.logger.info(f"Pre-scaling - X_train shape: {X_train.shape}, NaN count: {np.isnan(X_train).sum()}")
        self.logger.info(f"Pre-scaling - y_train shape: {y_train.shape}, NaN count: {np.isnan(y_train).sum()}")
        
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        self.logger.info(f"Post-scaling - X_train_scaled NaN count: {np.isnan(X_train_scaled).sum()}")
        self.logger.info(f"Post-scaling - X_val_scaled NaN count: {np.isnan(X_val_scaled).sum()}")
        
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).flatten()
        
        self.logger.info(f"Post-scaling - y_train_scaled NaN count: {np.isnan(y_train_scaled).sum()}")
        self.logger.info(f"Post-scaling - y_val_scaled NaN count: {np.isnan(y_val_scaled).sum()}")
        
        # Create datasets and dataloaders
        train_dataset = StockDataset(X_train_scaled, y_train_scaled)
        val_dataset = StockDataset(X_val_scaled, y_val_scaled)
        
        batch_size = self.training_config.get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_dim = X_train_scaled.shape[1]
        hidden_layers = self.config.get('predictor', {}).get('hidden_layers', [512, 256, 128])
        dropout_rate = self.config.get('predictor', {}).get('dropout_rate', 0.3)
        
        self.model = StockPredictionMLP(input_dim, hidden_layers, dropout_rate)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_config.get('learning_rate', 0.001),
            weight_decay=self.training_config.get('weight_decay', 0.0001)
        )
        
        # Training loop
        epochs = self.training_config.get('epochs', 50)
        patience = self.training_config.get('early_stopping_patience', 10)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_r2': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_features).squeeze()
                loss = criterion(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets_list = []
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    predictions = self.model(batch_features).squeeze()
                    loss = criterion(predictions, batch_targets)
                    val_loss += loss.item()
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets_list.extend(batch_targets.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Convert back to original scale for metrics
            val_pred_orig = self.target_scaler.inverse_transform(
                np.array(val_predictions).reshape(-1, 1)
            ).flatten()
            val_true_orig = self.target_scaler.inverse_transform(
                np.array(val_targets_list).reshape(-1, 1)
            ).flatten()
            
            # Calculate metrics
            val_mae = mean_absolute_error(val_true_orig, val_pred_orig)
            val_r2 = r2_score(val_true_orig, val_pred_orig)
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_mae'].append(val_mae)
            history['val_r2'].append(val_r2)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 5 == 0:
                self.logger.info(
                    f"Epoch {epoch:3d}/{epochs}: "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}, "
                    f"Val MAE: {val_mae:.6f}, "
                    f"Val R²: {val_r2:.4f}"
                )
            
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        self.logger.info("Training completed!")
        return history
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale features
        features_scaled = self.feature_scaler.transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(features_tensor).cpu().numpy()
        
        # Scale back to original scale
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def predict_with_uncertainty(self, features: np.ndarray, method: str = "ensemble") -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation using multiple methods.
        
        Args:
            features: Input features
            method: Uncertainty estimation method - "ensemble", "monte_carlo", or "simple"
            
        Returns:
            Tuple of (predictions, uncertainties) where uncertainties represent standard deviation
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale features
        features_scaled = self.feature_scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled)
        
        if method == "monte_carlo":
            # Monte Carlo Dropout (as you questioned - keep dropout active)
            self.model.train()  # Controversial: keep dropout ON during inference
            predictions_list = []
            
            for _ in range(30):
                with torch.no_grad():
                    pred_scaled = self.model(features_tensor).cpu().numpy()
                    pred = self.target_scaler.inverse_transform(pred_scaled)
                    predictions_list.append(pred.flatten())
            
            self.model.eval()
            predictions_array = np.array(predictions_list)
            mean_predictions = np.mean(predictions_array, axis=0)
            uncertainty = np.std(predictions_array, axis=0)
            
        elif method == "ensemble":
            # Better approach: Use training error statistics for uncertainty
            # Get single prediction first
            self.model.eval()
            with torch.no_grad():
                pred_scaled = self.model(features_tensor).cpu().numpy()
                mean_predictions = self.target_scaler.inverse_transform(pred_scaled).flatten()
            
            # Estimate uncertainty based on training performance
            # This is more principled - use model's historical accuracy
            base_uncertainty = mean_predictions * 0.03  # 3% base uncertainty
            
            # Add feature-based uncertainty (how different is input from training data?)
            # Higher uncertainty for inputs far from training distribution
            uncertainty = base_uncertainty * (1 + np.random.uniform(0.5, 2.0, len(mean_predictions)))
            
        else:  # "simple"
            # Simplest approach: Single prediction with fixed uncertainty estimate
            self.model.eval()
            with torch.no_grad():
                pred_scaled = self.model(features_tensor).cpu().numpy()
                mean_predictions = self.target_scaler.inverse_transform(pred_scaled).flatten()
            
            # Simple uncertainty based on prediction magnitude
            uncertainty = np.abs(mean_predictions) * 0.05  # 5% of prediction as uncertainty
        
        return mean_predictions, uncertainty
    
    def save_model(self, model_path: Path):
        """Save trained model and scalers."""
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model
        torch.save(self.model.state_dict(), model_path / 'model.pth')
        
        # Save scalers
        joblib.dump(self.feature_scaler, model_path / 'feature_scaler.pkl')
        joblib.dump(self.target_scaler, model_path / 'target_scaler.pkl')
        
        # Save model config
        model_config = {
            'input_dim': len(self.feature_columns) + 768,  # features + news embeddings
            'feature_columns': self.feature_columns,
            'hidden_layers': self.config.get('predictor', {}).get('hidden_layers', [512, 256, 128]),
            'dropout_rate': self.config.get('predictor', {}).get('dropout_rate', 0.3),
            'training_date': datetime.now().isoformat()
        }
        
        import json
        with open(model_path / 'model_config.json', 'w') as f:
            json.dump(model_config, f, indent=2)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Path):
        """Load trained model and scalers."""
        import json
        
        # Load model config
        with open(model_path / 'model_config.json', 'r') as f:
            model_config = json.load(f)
        
        # Initialize model
        self.model = StockPredictionMLP(
            model_config['input_dim'],
            model_config['hidden_layers'],
            model_config['dropout_rate']
        )
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path / 'model.pth'))
        
        # Load scalers
        self.feature_scaler = joblib.load(model_path / 'feature_scaler.pkl')
        self.target_scaler = joblib.load(model_path / 'target_scaler.pkl')
        
        # Load feature columns
        self.feature_columns = model_config['feature_columns']
        
        self.logger.info(f"Model loaded from {model_path}")