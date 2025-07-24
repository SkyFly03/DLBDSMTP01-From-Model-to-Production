# run_pipeline.ps1
# ---------------------------------------------------------
# Executes the full image classification pipeline in PowerShell
# including preprocessing, model training, batch predictions,
# and result visualization (shell-trained version).
# ---------------------------------------------------------

Write-Host "Activating virtual environment..."
& .\venv\Scripts\Activate.ps1

Write-Host "`nRunning data preparation steps..."
python organize_by_class.py
python sync_class_folders.py

Write-Host "`n[SKIPPED] Training model (already completed)"
# python train_model_full_set.py

Write-Host "`nStarting Docker containers..."
docker-compose up -d

Write-Host "`nRunning batch predictions..."
python batch_predict.py

Write-Host "`nGenerating training performance visualizations..."
python visualize_training_results.py

Write-Host "`nGenerating confusion matrix for shell-trained model..."
python confusion_matrix_heatmap2_shell.py

Write-Host "`nPipeline complete!"
