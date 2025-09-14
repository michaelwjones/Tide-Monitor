Write-Host "Transformer v2 Data Preparation Pipeline"
Write-Host "========================================="

Write-Host ""
Write-Host "Step 1: Fetching Firebase data..."
python fetch_firebase_data.py

Write-Host ""
Write-Host "Step 2: Creating training data..."
python create_training_data.py

Write-Host ""
Write-Host "Data preparation complete!"
Write-Host "Training files should be available in the data/ directory"