import torch
import torch.nn as nn
import math

class TidalTransformer(nn.Module):
    """
    Encoder-only transformer for tidal prediction.
    
    Architecture:
    - Encoder-only transformer (no decoder)
    - Learnable positional encoding
    - Input: 433 time steps (72 hours at 10-minute intervals)
    - Output: 144 time steps (24 hours at 10-minute intervals)
    """
    
    def __init__(self, input_dim=1, d_model=512, nhead=8, num_layers=6, 
                 output_dim=144, dropout=0.1, max_seq_length=433):
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Input projection: map from input_dim to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Learnable positional encoding (matches single run model)
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_length, d_model))
        
        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection: map from d_model to output_dim
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the transformer.
        
        Args:
            x: Input sequence (batch_size, seq_length, input_dim)
            
        Returns:
            output: Predicted sequence (batch_size, output_dim)
        """
        # x shape: (batch_size, seq_length, input_dim)
        seq_length = x.size(1)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_length].unsqueeze(0)
        x = self.dropout(x)
        
        # Apply transformer encoder
        x = self.transformer(x)
        
        # Use last token for prediction (matches single run approach)
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Project to output
        output = self.output_projection(x)
        return output
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'Encoder-Only Transformer',
            'encoder_layers': self.num_layers,
            'attention_heads': self.nhead,
            'hidden_dimension': self.d_model,
            'dropout': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_sequence_length': 433,
            'output_sequence_length': 144,
            'interval_minutes': 10,
            'positional_encoding': 'Learnable'
        }

def create_model(d_model=512, nhead=16, num_layers=4, dropout=0.17305796046740565):
    """Factory function to create TidalTransformer with single run configuration
    
    Args:
        d_model: Hidden dimension size (default: 512 from single run)
        nhead: Number of attention heads (default: 16 from single run) 
        num_layers: Number of encoder layers (default: 4 from single run)
        dropout: Dropout probability (default: 0.173 from single run)
    """
    return TidalTransformer(
        input_dim=1,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_dim=144,  # 24 hours at 10-minute intervals
        dropout=dropout,
        max_seq_length=433
    )

if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing TidalTransformer model (Encoder-Only)...")
    
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
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        output = model(src)
    print(f"Output shape: {output.shape} (24h at 10-min intervals)")
    
    print("Model test completed successfully!")
    
    # Test loading saved checkpoint
    try:
        checkpoint = torch.load('best.pth', map_location='cpu')
        print(f"\nCheckpoint loaded successfully!")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"Saved config: {config}")
            
            # Create model with saved config
            model_from_checkpoint = create_model(
                d_model=config.get('d_model', 512),
                nhead=config.get('nhead', 16),
                num_layers=config.get('num_layers', 4),
                dropout=config.get('dropout', 0.173)
            )
            
            # Load state dict
            model_from_checkpoint.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded from checkpoint successfully!")
            
    except FileNotFoundError:
        print("\nNo checkpoint file found (best.pth)")
    except Exception as e:
        print(f"\nError loading checkpoint: {e}")