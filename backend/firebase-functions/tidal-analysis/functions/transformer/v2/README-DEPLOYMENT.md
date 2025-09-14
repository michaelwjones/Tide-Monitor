# Transformer v2 Deployment Guide

This guide explains the two-step deployment process for transformer v2 inference to Firebase Functions.

## Overview

Due to the large model file size (306MB), the deployment is split into two steps:
1. **Upload model to Firebase Storage** - Stores the large model files in cloud storage
2. **Deploy inference function** - Deploys the function code that downloads models on demand

## Prerequisites

1. **Google Cloud SDK** installed and authenticated:
   ```bash
   # Install from: https://cloud.google.com/sdk/docs/install
   gcloud auth login
   gcloud config set project tide-monitor-boron
   ```

2. **Firebase CLI** installed and authenticated:
   ```bash
   npm install -g firebase-tools
   firebase login
   ```

3. **Firebase Storage** enabled in Firebase Console
4. **Trained Model** available in `shared/model.pth` (from training process)

## Deployment Process

The scripts are now pre-configured for your Firebase Storage bucket (`tide-monitor-boron.firebasestorage.app`).

## Step 1: Upload Model to Firebase Storage

Run the model upload script:

```powershell
.\upload-transformer-v2-model.ps1
```

This script:
- Verifies model files exist
- Uploads `model.pth` (306MB) to Firebase Storage
- Uploads `normalization_params.json` to Firebase Storage
- Sets public read permissions
- Shows download URLs and file sizes

**Storage locations:**
- `gs://tide-monitor-boron.firebasestorage.app/transformer-v2-models/model.pth`
- `gs://tide-monitor-boron.firebasestorage.app/transformer-v2-models/normalization_params.json`

## Step 2: Deploy Inference Function

Run the inference deployment script:

```powershell
.\deploy-transformer-v2-inference.ps1
```

This script:
- Verifies code files exist
- Checks if model was uploaded to Storage
- Copies shared code to inference directory
- Deploys function to Firebase Functions
- Creates `run_transformer_v2_analysis` scheduled function

## Function Behavior

### First Execution (Cold Start)
1. Function starts up
2. Downloads model from Firebase Storage to `/tmp/transformer-v2-model.pth`
3. Downloads normalization params to `/tmp/transformer-v2-normalization_params.json`  
4. Loads model and runs inference
5. **Duration**: ~60-90 seconds (includes 30-60s download time)

### Subsequent Executions (Warm Start)
1. Function starts up
2. Uses cached model from `/tmp/` directory
3. Loads model and runs inference
4. **Duration**: ~10-30 seconds (no download needed)

## Firebase Paths

### Forecast Data
- **Path**: `/tidal-analysis/transformer-v2-forecast`
- **Updates**: Every 5 minutes
- **Contents**: 144 predictions (24 hours at 10-minute intervals)

### Error Logging
- **Path**: `/tidal-analysis/transformer-v2-error`
- **Updates**: When errors occur
- **Contents**: Error details and timestamps

## Monitoring

### Function Logs
```bash
firebase functions:log --only run_transformer_v2_analysis
```

### Storage Usage
Check Firebase Console > Storage for uploaded models

### Function Metrics  
Check Firebase Console > Functions for execution metrics

## Troubleshooting

### Model Upload Issues
- Ensure Google Cloud SDK is installed and authenticated
- Check project permissions for Firebase Storage
- Verify model file exists and isn't corrupted

### Deployment Issues  
- Ensure Firebase CLI is installed and authenticated
- Check that virtual environment exists in inference directory
- Verify model was uploaded to Storage first

### Runtime Issues
- Check function logs for download errors
- Monitor function memory usage (2048MB allocated)
- Verify Storage permissions allow public read access

## File Structure

```
transformer/v2/
├── shared/
│   ├── model.pth                    # 306MB trained model
│   ├── model.py                     # Model architecture
│   ├── inference.py                 # Inference engine
│   └── normalization_params.json   # Data normalization params
├── inference/
│   ├── main.py                      # Firebase Functions runtime
│   ├── requirements.txt             # Python dependencies  
│   ├── firebase.json                # Deployment configuration
│   └── .firebaserc                  # Project configuration
├── upload-transformer-v2-model.ps1      # Model upload script
├── deploy-transformer-v2-inference.ps1  # Function deployment script
└── README-DEPLOYMENT.md                 # This file
```

## Cost Considerations

- **Storage**: ~$0.020/month for 306MB model file
- **Downloads**: ~$0.12 per GB egress (first function execution per instance)
- **Functions**: Standard Firebase Functions pricing for 2GB memory, 540s timeout
- **Caching**: Subsequent executions use cached model (no additional download costs)

## Security

- Model files are stored with public read access for function download
- Function uses default Firebase service account for Storage access
- No sensitive data is stored in model files (only neural network weights)