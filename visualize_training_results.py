# visualize_training_results.py
# ---------------------------------------------------------
# Visualizes MLflow-logged metrics: training/validation loss & accuracy.
# ---------------------------------------------------------
import mlflow
import matplotlib.pyplot as plt

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("Refund_Classifier")

if experiment is None:
    raise Exception("No MLflow experiment named 'Refund_Classifier' found.")

runs = client.search_runs(experiment.experiment_id)
latest_run = sorted(runs, key=lambda r: r.start_time, reverse=True)[0]

metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
steps = []

for metric in metrics.keys():
    history = client.get_metric_history(latest_run.info.run_id, metric)
    history = sorted(history, key=lambda x: x.step)
    metrics[metric] = [point.value for point in history]
    if not steps:
        steps = [point.step for point in history]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(steps, metrics["train_acc"], label="Train Accuracy")
plt.plot(steps, metrics["val_acc"], label="Val Accuracy")
plt.plot(steps, metrics["train_loss"], label="Train Loss", linestyle="--")
plt.plot(steps, metrics["val_loss"], label="Val Loss", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Performance (from MLflow)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model_images/training_metrics.png")
print("Saved: model_images/training_metrics.png")
