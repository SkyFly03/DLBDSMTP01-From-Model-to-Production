#!/bin/bash

echo "Activating virtual environment..."
source ./venv/Scripts/activate

echo "Running data preparation steps..."
python organize_by_class.py
python sync_class_folders.py

echo "Training model..."
python train_model_full_set.py

echo "Starting Docker containers..."
docker compose up -d

echo "Running batch predictions..."
python batch_predict.py

echo "Generating visualizations..."
python visualize_training_results.py
python confusion_matrix_heatmap.py

echo "Pipeline complete!"

