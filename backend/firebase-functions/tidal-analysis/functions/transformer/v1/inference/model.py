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
    Sequence-to-sequence transformer for tidal prediction with 10-minute intervals.
    
    Architecture:
    - 6-layer encoder, 3-layer decoder
    - 8 attention heads, 256 hidden dimensions
    - Input: 433 time steps (72 hours at 10-minute intervals)
    - Output: 144 time steps (24 hours at 10-minute intervals)
    """
    
    def __init__(self, 
                 input_dim=1,
                 d_model=256,
                 nhead=8, 
                 num_encoder_layers=6,
                 num_decoder_layers=3,
                 dim_feedforward=1024,
                 dropout=0.1,
                 max_seq_length=5000):
        super(TidalTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input projection: map from input_dim to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Output projection: map from d_model back to input_dim
        self.output_projection = nn.Linear(d_model, input_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer architecture
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (batch_size, seq_len, d_model) for better performance
        )
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.transformer, 'enable_nested_tensor'):
            self.transformer.enable_nested_tensor = False
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights using Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def generate_square_subsequent_mask(self, sz: int):
        """Generate causal mask for decoder to prevent looking at future tokens"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def forward(self, src, tgt=None, teacher_forcing_ratio=1.0):
        """
        Forward pass through the transformer.
        
        Args:
            src: Source sequence (batch_size, input_seq_len, input_dim)
            tgt: Target sequence for teacher forcing (batch_size, output_seq_len, input_dim)
            teacher_forcing_ratio: Probability of using teacher forcing vs autoregressive
            
        Returns:
            output: Predicted sequence (batch_size, output_seq_len, input_dim)
        """
        batch_size, src_seq_len, _ = src.shape
        
        # Project input to model dimension (batch_first=True, no transpose needed)
        src = self.input_projection(src)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding (need to transpose for pos_encoder, then back)
        src = src.transpose(0, 1)         # (seq_len, batch_size, d_model) for pos_encoder
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)         # (batch_size, seq_len, d_model) for transformer

        if self.training and tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # Training with teacher forcing
            tgt = self.input_projection(tgt)  # (batch_size, seq_len, d_model)
            
            # Add positional encoding (transpose for pos_encoder, then back)
            tgt = tgt.transpose(0, 1)         # (seq_len, batch_size, d_model) for pos_encoder
            tgt = self.pos_encoder(tgt)
            tgt = tgt.transpose(0, 1)         # (batch_size, seq_len, d_model) for transformer
            
            # Create causal mask for decoder
            tgt_seq_len = tgt.size(1)  # seq_len dimension is now index 1
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
            
            # Transformer forward pass
            output = self.transformer(src, tgt, tgt_mask=tgt_mask)
            
        else:
            # Inference or training without teacher forcing (autoregressive)
            output_seq_len = 144  # 24 hours at 10-minute intervals
            
            # Start with encoder output
            memory = self.transformer.encoder(src)
            
            # Memory-efficient autoregressive generation with sliding window
            max_context = 72  # Keep only last 72 tokens for context
            decoder_input = torch.zeros(batch_size, 1, self.d_model).to(src.device)  # (batch_size, 1, d_model)
            outputs = []
            
            # Autoregressive generation
            for i in range(output_seq_len):
                # Always slice to max_context to avoid conditional logic during tracing
                # This is safe even when current_len <= max_context
                decoder_input = decoder_input[:, -max_context:, :]
                current_len = decoder_input.size(1)
                
                # Use fixed window size to avoid dynamic operations during tracing
                # Use current_len directly since it's now bounded by max_context
                tgt_mask = self.generate_square_subsequent_mask(current_len).to(src.device)
                
                # Decoder forward pass with full decoder input (already windowed above)
                decoder_output = self.transformer.decoder(
                    decoder_input, memory, tgt_mask=tgt_mask
                )
                
                # Take the last output token
                current_output = decoder_output[:, -1:, :]  # (batch_size, 1, d_model)
                outputs.append(current_output.clone())  # Clone to avoid memory references
                
                # Append current output to decoder input
                decoder_input = torch.cat([decoder_input, current_output], dim=1)  # Concat along seq_len
                
                # Clear cache periodically during inference
                if i % 100 == 0 and not torch.jit.is_tracing():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Concatenate all outputs
            output = torch.cat(outputs, dim=1)  # (batch_size, output_seq_len, d_model)
        
        # Project back to input dimension (already in batch_first format)
        output = self.output_projection(output)  # (batch_size, seq_len, input_dim)
        
        return output
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'Seq2Seq Transformer',
            'encoder_layers': 6,
            'decoder_layers': 3,
            'attention_heads': 8,
            'hidden_dimension': self.d_model,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_sequence_length': 433,
            'output_sequence_length': 144,
            'interval_minutes': 10
        }

def create_model():
    """Factory function to create TidalTransformer with default configuration"""
    return TidalTransformer(
        input_dim=1,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1
    )

if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing TidalTransformer model...")
    
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
    tgt = torch.randn(batch_size, 144, 1)  # 24 hours target at 10-minute intervals
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: {src.shape} (72h at 10-min intervals)")
    print(f"Target shape: {tgt.shape} (24h at 10-min intervals)")
    
    # Test with teacher forcing
    model.train()
    output = model(src, tgt)
    print(f"Output shape (training): {output.shape}")
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        output = model(src)
    print(f"Output shape (inference): {output.shape}")
    
    print("Model test completed successfully!")