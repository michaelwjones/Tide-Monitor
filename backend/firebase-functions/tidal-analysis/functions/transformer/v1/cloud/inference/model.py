import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer sequences.
    Adds positional information to embeddings without learnable parameters.
    """
    
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to input embeddings
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

class TidalTransformer(nn.Module):
    """
    Single-pass encoder-only transformer for tidal prediction.
    
    Architecture:
    - Transformer encoder with multi-head attention
    - Input: 433 time steps (72 hours at 10-minute intervals)
    - Output: 144 time steps (24 hours at 10-minute intervals)
    - Single forward pass for both training and inference
    """
    
    def __init__(self, 
                 input_dim=1,
                 d_model=256,
                 nhead=8, 
                 num_encoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 max_seq_length=5000):
        super(TidalTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.input_seq_len = 433
        self.output_seq_len = 144
        
        # Input projection: map from input_dim to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        
        # Simple approach: use evenly spaced positions from the encoded sequence
        # 433 input steps -> select every ~3rd position to get 144 outputs
        output_positions = torch.linspace(0, 432, 144).long()
        self.register_buffer('output_positions', output_positions)
        
        # Direct prediction head for each selected position
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights using Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, src):
        """
        Single-pass forward through the transformer.
        
        Args:
            src: Source sequence (batch_size, input_seq_len, input_dim)
            
        Returns:
            output: Predicted sequence (batch_size, output_seq_len, input_dim)
        """
        batch_size, src_seq_len, _ = src.shape
        
        # Ensure input is the expected length
        if src_seq_len != self.input_seq_len:
            raise ValueError(f"Expected input length {self.input_seq_len}, got {src_seq_len}")
        
        # Project input to model dimension
        src = self.input_projection(src)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding (need to transpose for pos_encoder, then back)
        src = src.transpose(0, 1)         # (seq_len, batch_size, d_model) for pos_encoder
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)         # (batch_size, seq_len, d_model) for transformer
        
        # Process through transformer encoder
        encoded = self.transformer_encoder(src)  # (batch_size, seq_len, d_model)
        
        # Select evenly spaced positions from the encoded sequence
        selected_positions = encoded[:, self.output_positions, :]  # (batch_size, 144, d_model)
        
        # Apply prediction head to each selected position
        output = self.prediction_head(selected_positions)  # (batch_size, 144, 1)
        
        return output
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'Single-Pass Encoder Transformer',
            'encoder_layers': self.num_encoder_layers,
            'attention_heads': self.nhead,
            'hidden_dimension': self.d_model,
            'dropout': self.dropout,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_sequence_length': self.input_seq_len,
            'output_sequence_length': self.output_seq_len,
            'interval_minutes': 10
        }

def create_model(d_model=256, nhead=8, num_layers=6, dropout=0.1):
    """Factory function to create TidalTransformer with configurable parameters
    
    Args:
        d_model: Hidden dimension size
        nhead: Number of attention heads
        num_layers: Number of encoder layers
        dropout: Dropout probability
    """
    return TidalTransformer(
        input_dim=1,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        dim_feedforward=d_model * 4,  # Standard transformer ratio
        dropout=dropout
    )

if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing Single-Pass TidalTransformer model...")
    
    model = create_model()
    print("Model created successfully!")
    
    # Print model info
    info = model.get_model_info()
    print(f"\nModel Information:")
    for key, value in info.items():
        if isinstance(value, int) and value > 1000:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    
    # Test forward pass
    batch_size = 2
    src = torch.randn(batch_size, 433, 1)  # 72 hours input at 10-minute intervals
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: {src.shape} (72h at 10-min intervals)")
    
    # Test training mode
    model.train()
    output = model(src)
    print(f"Output shape (training): {output.shape}")
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        output = model(src)
    print(f"Output shape (inference): {output.shape}")
    
    print("✅ Single-pass transformer test completed successfully!")
    print("✅ Same behavior in training and inference - no autoregressive generation!")