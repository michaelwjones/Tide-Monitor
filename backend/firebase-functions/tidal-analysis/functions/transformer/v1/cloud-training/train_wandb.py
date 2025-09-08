"""
W&B cloud training script for Transformer v1 tidal prediction.
Fetches data from Firebase, trains model, and tracks experiments.
"""
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import sys
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))

from model import create_model
from dataset import TidalDataset

def fetch_firebase_data():
    """Fetch training data directly from Firebase"""
    print("Fetching data from Firebase...")
    
    # Initialize Firebase (you'll need your service account key)
    if not firebase_admin._apps:
        # Use environment variable or upload service account key
        cred = credentials.Certificate("path/to/serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://tide-monitor-boron-default-rtdb.firebaseio.com'
        })
    
    # Fetch readings
    ref = db.reference('readings')
    readings = ref.order_by_key().get()
    
    print(f"Fetched {len(readings)} readings from Firebase")
    return readings

def prepare_data_for_training(readings):
    """Convert Firebase readings to training sequences"""
    # This would contain the same logic as your data-preparation scripts
    # but adapted for cloud execution
    pass

def train_with_wandb():
    """Main training function with W&B integration"""
    
    # Initialize W&B
    wandb.init(
        project="tide-transformer-v1",
        config={
            "batch_size": 4,
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "model_type": "seq2seq_transformer",
            "input_length": 433,
            "output_length": 144
        }
    )
    
    config = wandb.config
    
    # Fetch and prepare data
    raw_data = fetch_firebase_data()
    # ... data preparation logic here
    
    # Create model
    model = create_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Training on device: {device}")
    
    # Training loop with W&B logging
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(config.num_epochs):
        # Training step
        model.train()
        train_loss = 0.0
        
        # ... your training loop here
        
        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch}: Loss = {train_loss:.6f}")
    
    # Save model artifact to W&B
    model_path = "transformer_v1_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': dict(config),
        'final_loss': train_loss
    }, model_path)
    
    wandb.save(model_path)
    print("Training complete!")

if __name__ == "__main__":
    train_with_wandb()