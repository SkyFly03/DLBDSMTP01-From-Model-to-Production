# visualize_training_results.py
# --------------------------------------------------
# Example script to extract and plot accuracy and loss
# from MLflow logs after training is complete.
# --------------------------------------------------

import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# Load latest MLflow run (now that mlruns is available)
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("Refund_Classifier")
runs = client.search_runs(experiment.experiment_id)

# Just use the first available run
run = runs[0]
run_id = run.info.run_id

# Retrieve metrics
metrics = client.get_metric_history(run_id, "train_acc")
train_acc = [m.value for m in metrics]
train_steps = [m.step for m in metrics]

metrics = client.get_metric_history(run_id, "val_acc")
val_acc = [m.value for m in metrics]

metrics = client.get_metric_history(run_id, "train_loss")
train_loss = [m.value for m in metrics]

metrics = client.get_metric_history(run_id, "val_loss")
val_loss = [m.value for m in metrics]

# Plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_steps, train_acc, label="Train Accuracy")
plt.plot(train_steps, val_acc, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_steps, train_loss, label="Train Loss")
plt.plot(train_steps, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

plt.tight_layout()
plt.show()
