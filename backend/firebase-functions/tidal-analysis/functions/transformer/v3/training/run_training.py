#!/usr/bin/env python3
"""
Training execution script for Transformer v2.
Loads prepared data and runs training on Modal with H100 GPU.
"""

import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import modal

def load_training_data():
    """Load prepared training data from data-preparation directory."""
    
    data_dir = Path("../data-preparation/data")
    
    print("Loading training data...")
    print(f"Data directory: {data_dir.absolute()}")
    
    # Check if all required files exist
    required_files = {
        'X_train.npy': 'Training input sequences',
        'y_train.npy': 'Training target sequences', 
        'X_val.npy': 'Validation input sequences',
        'y_val.npy': 'Validation target sequences',
        'metadata.json': 'Dataset metadata',
        'normalization_params.json': 'Normalization parameters'
    }
    
    missing_files = []
    for filename in required_files:
        if not (data_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print("ERROR: Missing required data files:")
        for filename in missing_files:
            print(f"  - {filename}")
        print("\nPlease run data preparation first:")
        print("  cd ../data-preparation")
        print("  python fetch_firebase_data.py")
        print("  python create_training_data.py")
        return None
    
    # Load data files
    data = {}
    
    print("\nLoading data files:")
    for filename, description in required_files.items():
        filepath = data_dir / filename
        print(f"  Loading {filename}... ", end="", flush=True)
        
        if filename.endswith('.npy'):
            array = np.load(filepath)
            data[filename] = array
            print(f"OK Shape: {array.shape}, Size: {array.nbytes / 1024 / 1024:.1f} MB")
        else:
            with open(filepath, 'r') as f:
                json_data = json.load(f)
            data[filename] = json_data
            print(f"OK Loaded")
    
    # Display data summary
    print(f"\nData Summary:")
    print(f"  Training sequences: {len(data['X_train.npy']):,}")
    print(f"  Validation sequences: {len(data['X_val.npy']):,}")
    print(f"  Input length: {data['X_train.npy'].shape[1]} time steps")
    print(f"  Output length: {data['y_train.npy'].shape[1]} time steps")
    print(f"  Data range: {data['X_train.npy'].min():.3f} to {data['X_train.npy'].max():.3f}")
    
    return data

def create_training_config():
    """Create training configuration."""
    
    config = {
        # Model configuration
        'model_config': {
            'input_length': 432,
            'output_length': 144,
            'd_model': 512,
            'nhead': 16,
            'num_encoder_layers': 8,
            'dim_feedforward': 2048,
            'dropout': 0.1
        },
        
        # Training configuration  
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'log_interval': 50,
        
        # Training metadata
        'experiment_name': f'transformer_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'description': 'Transformer v2 baseline training with H100 GPU'
    }
    
    return config

def run_training():
    """Main function to execute training on Modal."""
    
    print("=" * 60)
    print("TRANSFORMER V2 TRAINING EXECUTION")
    print("=" * 60)
    print()
    
    # Load training data
    data = load_training_data()
    if data is None:
        return False
    
    # Create training configuration
    config = create_training_config()
    
    print(f"\nTraining Configuration:")
    print(f"  Experiment: {config['experiment_name']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Model dimension: {config['model_config']['d_model']}")
    print(f"  Encoder layers: {config['model_config']['num_encoder_layers']}")
    print(f"  Attention heads: {config['model_config']['nhead']}")
    
    # Connect to deployed Modal app
    print(f"\nConnecting to Modal...")
    try:
        import subprocess
        
        # Check that Modal CLI is working
        result = subprocess.run(["python", "-m", "modal", "app", "list"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Modal CLI not working")
        
        print("OK Modal CLI available")
    except Exception as e:
        print(f"ERROR: Failed to access Modal: {e}")
        return False
    
    # Prepare data for Modal
    print(f"\nPreparing data for training...")
    train_data_x = data['X_train.npy']
    train_data_y = data['y_train.npy'] 
    val_data_x = data['X_val.npy']
    val_data_y = data['y_val.npy']
    
    print(f"  Training data: {train_data_x.shape} -> {train_data_y.shape}")
    print(f"  Validation data: {val_data_x.shape} -> {val_data_y.shape}")
    print(f"  Total data size: {(train_data_x.nbytes + train_data_y.nbytes + val_data_x.nbytes + val_data_y.nbytes) / 1024 / 1024:.1f} MB")
    
    # Start training
    print(f"\n" + "=" * 60)
    print("STARTING TRAINING ON MODAL H100 GPU")
    print("=" * 60)
    print()
    print("Training will run on Modal's cloud infrastructure.")
    print("You can monitor progress in the console output below.")
    print("Training artifacts will be saved to Modal volume.")
    print()
    
    try:
        # Save training data to files for the modal run
        print(f"Preparing data files for Modal execution...")
        
        # Save data to the data directory (create temp data files)
        temp_data_path = Path("./temp_training_data")
        temp_data_path.mkdir(exist_ok=True)
        
        np.save(temp_data_path / "X_train.npy", train_data_x)
        np.save(temp_data_path / "y_train.npy", train_data_y)
        np.save(temp_data_path / "X_val.npy", val_data_x)
        np.save(temp_data_path / "y_val.npy", val_data_y)
        
        with open(temp_data_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"OK Data saved to {temp_data_path}")
        
        # Use modal run to execute the training function
        print(f"Executing training via Modal CLI...")
        
        cmd = [
            "python", "-m", "modal", "run", 
            "modal_setup.py",
            "--x-train-path", str(temp_data_path / "X_train.npy"),
            "--y-train-path", str(temp_data_path / "y_train.npy"), 
            "--x-val-path", str(temp_data_path / "X_val.npy"),
            "--y-val-path", str(temp_data_path / "y_val.npy"),
            "--config-path", str(temp_data_path / "config.json")
        ]
        
        # Set environment to handle Unicode properly
        import os
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        
        # Run the command and stream output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1, encoding='utf-8',
                                 env=env, errors='replace')
        
        print("Modal training started...")
        print("-" * 60)
        
        # Stream output in real-time with robust Unicode handling
        for line in process.stdout:
            try:
                print(line, end='')
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Replace any problematic Unicode characters
                safe_line = line.encode('ascii', 'replace').decode('ascii')
                print(safe_line, end='')
            except Exception:
                # Fallback for any other encoding issues
                print("[Output contained unreadable characters]")
                continue
        
        process.wait()
        
        # Clean up temp data
        import shutil
        shutil.rmtree(temp_data_path)
        
        if process.returncode == 0:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"\nTraining artifacts saved to Modal volume 'transformer-v2-storage'")
            return True
        else:
            print(f"\n" + "=" * 60)
            print("TRAINING FAILED")
            print("=" * 60)
            print(f"Process exited with code: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"\n" + "=" * 60)
        print("TRAINING FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        return False

def main():
    """Entry point."""
    success = run_training()
    
    if success:
        print(f"\nTraining completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Download trained model from Modal volume")  
        print(f"2. Evaluate model performance")
        print(f"3. Deploy for inference")
    else:
        print(f"\nTraining failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())