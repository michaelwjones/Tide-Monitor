import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer input sequences.
    Enables the model to understand temporal relationships in the data.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with multi-head attention and feed-forward network.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention block
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward block
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class TidalTransformerV2(nn.Module):
    """
    Transformer v2 for tidal water level prediction.
    
    Architecture:
    - Encoder-only transformer (single-pass prediction)
    - 8 encoder layers, 16 attention heads, 512 model dimension
    - Input: 432 time steps (72 hours at 10-minute intervals)
    - Output: 144 time steps (24 hours at 10-minute intervals)
    - Single feature: water level in mm (normalized)
    """
    
    def __init__(
        self,
        input_length=432,
        output_length=144,
        d_model=512,
        nhead=16,
        num_encoder_layers=8,
        dim_feedforward=2048,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_length = input_length
        self.output_length = output_length
        self.d_model = d_model
        
        # Input projection: convert single feature to model dimension
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_length + 50)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Output projection: convert encoded sequence to predictions
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_length),
            nn.Linear(output_length, output_length)  # Final refinement layer
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src):
        """
        Forward pass: single-pass prediction from input sequence to output sequence.
        
        Args:
            src: Input tensor of shape [batch_size, input_length, 1]
            
        Returns:
            predictions: Output tensor of shape [batch_size, output_length]
        """
        # Input validation
        batch_size, seq_len, feature_dim = src.shape
        assert seq_len == self.input_length, f"Expected input length {self.input_length}, got {seq_len}"
        assert feature_dim == 1, f"Expected 1 feature dimension, got {feature_dim}"
        
        # Project input to model dimension: [batch_size, seq_len, d_model]
        src = self.input_projection(src)
        
        # Add positional encoding
        src = src.transpose(0, 1)  # [seq_len, batch_size, d_model] for pos encoding
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # Back to [batch_size, seq_len, d_model]
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            src = layer(src)
        
        # Global average pooling across sequence dimension
        # This creates a fixed-size representation from variable-length sequences
        encoded = src.mean(dim=1)  # [batch_size, d_model]
        
        # Project to output predictions
        predictions = self.output_projection(encoded)  # [batch_size, output_length]
        
        return predictions
    
    def get_model_info(self):
        """Return model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'TidalTransformerV2',
            'input_length': self.input_length,
            'output_length': self.output_length,
            'd_model': self.d_model,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'encoder-only-transformer',
            'prediction_method': 'single-pass'
        }

def create_model(config=None):
    """
    Factory function to create TidalTransformerV2 model with default or custom configuration.
    
    Args:
        config: Optional dictionary with model configuration parameters
        
    Returns:
        model: TidalTransformerV2 instance
    """
    default_config = {
        'input_length': 432,
        'output_length': 144,
        'd_model': 512,
        'nhead': 16,
        'num_encoder_layers': 8,
        'dim_feedforward': 2048,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    model = TidalTransformerV2(**default_config)
    
    print(f"Created TidalTransformerV2 with {model.get_model_info()['total_parameters']:,} parameters")
    return model

if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_model()
    print("\nModel Info:")
    for key, value in model.get_model_info().items():
        print(f"  {key}: {value}")
    
    # Test forward pass with sample data
    batch_size = 4
    sample_input = torch.randn(batch_size, 432, 1)
    
    with torch.no_grad():
        output = model(sample_input)
        print(f"\nTest forward pass:")
        print(f"  Input shape: {sample_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")