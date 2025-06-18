import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

class StockPriceLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, input_dim)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # apply attention
        context, attention_weights = self.attention(lstm_out)
        
        # final prediction
        prediction = self.fc(context)
        
        return prediction, attention_weights

class StockPricePredictor:
    def __init__(
        self,
        input_dim: int = 8, 
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model = StockPriceLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout
        ).to(device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def prepare_sequence(self, data: Dict[str, Any], seq_length: int = 60) -> torch.Tensor:
        """
        Prepare input sequence from raw data.
        
        Args:
            data: Dictionary containing:
                - prices: List of historical prices
                - volume: List of trading volumes
                - rsi: List of RSI values
                - ma: List of moving averages
                - sentiment: List of sentiment scores
                - open: List of opening prices
                - high: List of high prices
                - low: List of low prices
            seq_length: Length of input sequence
            
        Returns:
            torch.Tensor: Prepared sequence of shape (1, seq_length, input_dim)
        """
        # Extract features
        features = []
        features.append(data['prices'][-seq_length:])  # close prices
        features.append(data['volume'][-seq_length:])
        features.append(data['rsi'][-seq_length:])
        features.append(data['ma'][-seq_length:])
        features.append(data['sentiment'][-seq_length:])
        features.append(data['open'][-seq_length:])
        features.append(data['high'][-seq_length:])
        features.append(data['low'][-seq_length:])
        
        # Stack features
        sequence = np.column_stack(features)
        
        # Normalize features with epsilon to prevent division by zero
        mean = np.mean(sequence, axis=0)
        std = np.std(sequence, axis=0)
        epsilon = 1e-8  # Small constant to prevent division by zero
        sequence = (sequence - mean) / (std + epsilon)
        
        # Convert to tensor
        sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        return sequence
    
    def predict(self, data: Dict[str, Any]) -> float:
        """
        Make a price prediction for the next time step.
        
        Args:
            data: Dictionary containing historical data
            
        Returns:
            float: Predicted price
        """
        self.model.eval()
        with torch.no_grad():
            sequence = self.prepare_sequence(data)
            prediction, _ = self.model(sequence)
            return prediction.item()
    
    def train_step(self, sequence: torch.Tensor, target: torch.Tensor) -> float:
        """
        Perform one training step.
        
        Args:
            sequence: Input sequence tensor
            target: Target price tensor
            
        Returns:
            float: Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        prediction, _ = self.model(sequence)
        loss = self.criterion(prediction, target)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item() 