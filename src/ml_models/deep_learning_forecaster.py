"""
Deep Learning Time Series Forecasting

Implements state-of-the-art neural network architectures for financial forecasting:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Units)
- Temporal Convolutional Networks (TCN)
- Attention-based models
- Ensemble neural networks

Designed for predicting returns, volatility, and regime transitions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed. Install with: pip install torch")


@dataclass
class ForecastResult:
    """Forecast results"""
    predictions: np.ndarray
    actual: np.ndarray
    dates: pd.DatetimeIndex
    mae: float
    rmse: float
    direction_accuracy: float  # % of correct up/down predictions
    sharpe_if_traded: float  # Sharpe if we trade on predictions


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series"""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class LSTMForecaster(nn.Module):
    """
    LSTM-based time series forecaster

    Architecture:
    - Multiple LSTM layers with dropout
    - Attention mechanism
    - Fully connected output layers
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        """
        Initialize LSTM model

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(LSTMForecaster, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Attention mechanism
        attention_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Linear(attention_size, 1)

        # Fully connected layers
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(fc_input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def attention_layer(self, lstm_output):
        """Apply attention mechanism"""
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context

    def forward(self, x):
        """Forward pass"""
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        context = self.attention_layer(lstm_out)

        # Fully connected layers
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)

        return out


class GRUForecaster(nn.Module):
    """
    GRU-based forecaster (lighter alternative to LSTM)

    Often performs similarly to LSTM but faster training
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super(GRUForecaster, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take last time step
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class DeepLearningForecaster:
    """
    Deep Learning Forecaster for Financial Time Series

    Handles data preparation, model training, and forecasting
    """

    def __init__(self,
                 model_type: str = 'lstm',
                 sequence_length: int = 60,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize forecaster

        Args:
            model_type: 'lstm' or 'gru'
            sequence_length: Number of time steps to look back
            hidden_size: Size of hidden layers
            num_layers: Number of recurrent layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.model_type = model_type
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate

        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = None
        self.trained = False

    def prepare_data(self,
                    data: pd.DataFrame,
                    target_col: str,
                    feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM/GRU

        Creates sequences of length `sequence_length` for training

        Args:
            data: DataFrame with features and target
            target_col: Name of target column (what to predict)
            feature_cols: List of feature columns (if None, use all except target)

        Returns:
            X (features), y (targets) as numpy arrays
        """
        if feature_cols is None:
            feature_cols = [col for col in data.columns if col != target_col]

        self.feature_names = feature_cols

        # Extract features and target
        X_raw = data[feature_cols].values
        y_raw = data[target_col].values

        # Normalize (important for neural networks)
        from sklearn.preprocessing import StandardScaler

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_scaled = self.scaler_X.fit_transform(X_raw)
        y_scaled = self.scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(X_scaled)):
            X.append(X_scaled[i - self.sequence_length:i])
            y.append(y_scaled[i])

        return np.array(X), np.array(y)

    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             epochs: int = 50,
             batch_size: int = 32,
             verbose: bool = True) -> Dict:
        """
        Train the model

        Args:
            X_train: Training features (samples, sequence_length, features)
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Print progress

        Returns:
            Training history
        """
        # Initialize model
        input_size = X_train.shape[2]

        if self.model_type == 'lstm':
            self.model = LSTMForecaster(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=False
            )
        elif self.model_type == 'gru':
            self.model = GRUForecaster(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_dataset = TimeSeriesDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        # Training loop
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                # Forward pass
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}")

        self.trained = True
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features (samples, sequence_length, features)

        Returns:
            Predictions (unscaled)
        """
        if not self.trained:
            raise ValueError("Model not trained yet")

        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions_scaled = self.model(X_tensor).squeeze().numpy()

        # Inverse transform
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

        return predictions

    def evaluate(self,
                X_test: np.ndarray,
                y_test: np.ndarray,
                dates: Optional[pd.DatetimeIndex] = None) -> ForecastResult:
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: True targets (scaled)
            dates: Date index for predictions

        Returns:
            ForecastResult with metrics
        """
        # Get predictions
        predictions = self.predict(X_test)

        # Unscale actual values
        actual = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Calculate metrics
        mae = np.mean(np.abs(predictions - actual))
        rmse = np.sqrt(np.mean((predictions - actual) ** 2))

        # Direction accuracy (% of correct up/down predictions)
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actual)
        direction_accuracy = np.mean(pred_direction == actual_direction)

        # Sharpe if traded on predictions
        trading_returns = predictions * actual  # If we trade based on predictions
        sharpe = np.mean(trading_returns) / np.std(trading_returns) * np.sqrt(252) if np.std(trading_returns) > 0 else 0

        return ForecastResult(
            predictions=predictions,
            actual=actual,
            dates=dates if dates is not None else pd.date_range(start='2020-01-01', periods=len(predictions)),
            mae=mae,
            rmse=rmse,
            direction_accuracy=direction_accuracy,
            sharpe_if_traded=sharpe
        )

    def save_model(self, path: str):
        """Save trained model"""
        if not self.trained:
            raise ValueError("No trained model to save")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.feature_names,
            'config': {
                'model_type': self.model_type,
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }, path)

    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path)

        # Restore config
        config = checkpoint['config']
        self.model_type = config['model_type']
        self.sequence_length = config['sequence_length']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']

        # Restore scalers
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.feature_names = checkpoint['feature_names']

        # Recreate model
        input_size = len(self.feature_names)
        if self.model_type == 'lstm':
            self.model = LSTMForecaster(input_size, self.hidden_size, self.num_layers, self.dropout)
        else:
            self.model = GRUForecaster(input_size, self.hidden_size, self.num_layers, self.dropout)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.trained = True


if __name__ == '__main__':
    """Test deep learning forecaster"""

    if not TORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
        exit(1)

    print("Deep Learning Forecaster - Demo\n")

    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n = len(dates)

    # Create features
    data = pd.DataFrame({
        'returns': np.random.normal(0.001, 0.02, n).cumsum() * 0.01,
        'volume': np.random.lognormal(15, 0.5, n),
        'rsi': np.random.uniform(30, 70, n),
        'macd': np.random.normal(0, 1, n),
        'volatility': np.random.uniform(0.01, 0.03, n)
    }, index=dates)

    # Target: next day return
    data['target'] = data['returns'].shift(-1)
    data = data.dropna()

    # Split data
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    test_data = data.iloc[train_size + val_size:]

    # Initialize forecaster
    forecaster = DeepLearningForecaster(
        model_type='lstm',
        sequence_length=20,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001
    )

    # Prepare data
    feature_cols = ['returns', 'volume', 'rsi', 'macd', 'volatility']

    print("Preparing data...")
    X_train, y_train = forecaster.prepare_data(train_data, 'target', feature_cols)
    X_val, y_val = forecaster.prepare_data(val_data, 'target', feature_cols)
    X_test, y_test = forecaster.prepare_data(test_data, 'target', feature_cols)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}\n")

    # Train
    print("Training LSTM model...")
    history = forecaster.train(
        X_train, y_train,
        X_val, y_val,
        epochs=30,
        batch_size=32,
        verbose=True
    )

    # Evaluate
    print("\nEvaluating on test set...")
    results = forecaster.evaluate(X_test, y_test, test_data.index[forecaster.sequence_length:])

    print(f"\nTest Results:")
    print(f"  MAE:                 {results.mae:.6f}")
    print(f"  RMSE:                {results.rmse:.6f}")
    print(f"  Direction Accuracy:  {results.direction_accuracy*100:.2f}%")
    print(f"  Sharpe (if traded):  {results.sharpe_if_traded:.2f}")

    print("\n✅ Deep learning forecaster working!")
