"""
Setup script for cloud training environment.
Uploads data and prepares W&B artifacts.
"""
import wandb
import json
import os

def create_data_artifact():
    """Create W&B artifact from local training data"""
    
    # Initialize W&B
    wandb.init(project="tide-transformer-v1", job_type="data-upload")
    
    # Create artifact
    artifact = wandb.Artifact("tidal-training-data", type="dataset")
    
    # Add training data files
    data_dir = "../data-preparation/data"
    if os.path.exists(data_dir):
        artifact.add_dir(data_dir, name="training_data")
        print("Added training data to artifact")
    else:
        print("Warning: Training data directory not found")
    
    # Add model files for reference
    model_dir = "../inference"
    if os.path.exists(model_dir):
        artifact.add_file(f"{model_dir}/model.py", name="model.py")
        print("Added model definition to artifact")
    
    # Log artifact
    wandb.log_artifact(artifact)
    print("Data artifact uploaded to W&B")
    
    wandb.finish()

def upload_service_account_key():
    """Instructions for uploading Firebase service account key"""
    print("""
To enable Firebase access in cloud training:

1. Download your Firebase service account key from:
   Firebase Console > Project Settings > Service Accounts > Generate new private key

2. Upload it to W&B as a secret:
   wandb login
   wandb artifact put serviceAccountKey.json --type secret

3. Update train_wandb.py to use the secret:
   artifact = wandb.use_artifact('serviceAccountKey.json:latest', type='secret')
   artifact_dir = artifact.download()
   
4. Or set as environment variable in W&B dashboard
""")

if __name__ == "__main__":
    print("Setting up cloud training environment...")
    create_data_artifact()
    upload_service_account_key()