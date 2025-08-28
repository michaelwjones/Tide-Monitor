import torch
import torch.nn as nn

class TidalLSTM(nn.Module):
    """
    LSTM model for single-step tidal prediction.
    
    Designed for training on 72-hour sequences to predict the next minute.
    During inference, predictions are fed back iteratively to generate 24-hour forecasts.
    """
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        super(TidalLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer for single-step prediction
        self.output_layer = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM weights using Xavier initialization"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0)
    
    def forward(self, x):
        """
        Forward pass for training.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output from sequence
        last_output = lstm_out[:, -1, :]
        
        # Single prediction
        prediction = self.output_layer(last_output)
        
        return prediction
    
    def predict_sequence(self, initial_sequence, steps=1440):
        """
        Generate multi-step predictions by feeding outputs back as inputs.
        Used during inference for iterative 24-hour forecasting.
        
        Args:
            initial_sequence: Initial sequence tensor (1, sequence_length, 1)
            steps: Number of prediction steps (1440 for 24 hours)
        
        Returns:
            predictions: List of predicted values
        """
        self.eval()
        predictions = []
        
        # Start with initial sequence
        current_sequence = initial_sequence.clone()
        
        with torch.no_grad():
            for _ in range(steps):
                # Predict next value
                prediction = self.forward(current_sequence)
                predictions.append(prediction.item())
                
                # Update sequence: remove first element, add prediction
                new_value = prediction.unsqueeze(0)  # Shape: (1, 1, 1)
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],  # Remove first timestep
                    new_value
                ], dim=1)
        
        return predictions
    
    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'TidalLSTM',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024  # Assuming float32
        }