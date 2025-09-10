# 🧠 Tide Transformer v1 - Modal Hyperparameter Optimization

Serverless GPU hyperparameter optimization using Modal and Ray Tune.

## Quick Start

### 1. Install Modal
```bash
pip install -r requirements.txt
modal token new
```

### 2. Upload Training Data
```bash
modal run modal_hp_sweep.py --upload-data E:\Projects\Tide-Monitor\backend\firebase-functions\tidal-analysis\functions\transformer\v1\data-preparation\data
```

### 3. Run Hyperparameter Sweep
```bash
modal run modal_hp_sweep.py --run-sweep
```

## What It Does

- **Serverless A100 GPUs**: Automatically spins up cloud GPUs as needed
- **Bayesian Optimization**: Smart hyperparameter search with BayesOpt + ASHA early stopping
- **20 Trials**: Tests 20 different hyperparameter combinations
- **Early Stopping**: Kills poor performers after 10 epochs to save time/money
- **Auto-scaling**: Runs up to 4 trials in parallel, scales to zero when done

## Hyperparameters Optimized

- **Architecture**: `d_model` (256-768), `nhead` (8-16), `num_layers` (4-10), `dropout` (0.1-0.4)
- **Training**: `learning_rate` (1e-5 to 1e-3), `batch_size` (16-40), `weight_decay` (1e-6 to 1e-4)

## Expected Performance

- **Duration**: 2-4 hours (T4 is slower but budget-friendly)
- **Cost**: $10-20 for full optimization
- **GPU**: NVIDIA T4 (16GB VRAM) - Perfect for first runs
- **Parallel Trials**: Up to 4 concurrent

## Results

Results saved to Modal volume at `/data/hp_optimization_results.json`:

```json
{
  "best_config": {
    "d_model": 512,
    "nhead": 12,
    "num_layers": 6,
    "dropout": 0.2,
    "learning_rate": 0.0003,
    "batch_size": 32,
    "weight_decay": 1e-5
  },
  "best_val_loss": 0.045,
  "top_5_configs": [...],
  "total_trials": 20
}
```

## Advanced Usage

### Monitor Progress
```bash
modal app list
modal app logs tide-transformer-v1-hp-sweep
```

### Download Results
```bash
modal volume get tide-training-data hp_optimization_results.json ./
```

### Custom Configuration
Edit `modal_hp_sweep.py` to modify:
- Search space ranges
- Number of trials
- GPU type (A100, H100)
- Optimization algorithm

## Architecture

- **Modal App**: `tide-transformer-v1-hp-sweep`
- **Volume**: `tide-training-data` (persistent storage)
- **Image**: Debian + PyTorch + Ray Tune
- **Resources**: A100 GPU, 16GB RAM, 2-hour timeout

## No W&B Required

Pure hyperparameter optimization with Ray Tune - no external experiment tracking needed. Results are saved locally and can be integrated with any ML pipeline.