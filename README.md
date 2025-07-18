# DLBDSMTP01 – From Model to Production  
## Task 2: Image Classification for a Refund Department (Batch Processing)

---

## **1. Objective**

- This project builds an image classification system to automate the sorting of refund items based on product images.  
- A ResNet50 model is trained to predict product categories.  
- The system supports both batch predictions (scheduled daily) and real-time requests via a REST API.  
- Prediction results are stored in a PostgreSQL database.  
- The setup is containerized using Docker for reproducibility and deployment.

---

## **2. Setup and Installation**

### Clone the repository
```bash
git clone https://github.com/SkyFly03/DLBDSMTP01-From-Model-to-Production.git
cd DLBDSMTP01-From-Model-to-Production
```

### Create and activate a virtual environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download dataset manually from Kaggle
- Source: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset  
- Place `styles.csv` in `data/full/`  

Final folder structure after processing:
```
data/full/train/     # Training images  
data/full/val/       # Validation images  
```

### Prepare dataset
- `make_full_set.py` — Filters and copies selected images  
- `prepare_full_dataset.py` — Splits data into train/val  
- `organize_by_class.py` — Sorts images by label  
- `sync_class_folders.py` — Ensures consistent category folders

### Start PostgreSQL and REST API
```bash
docker-compose up -d
```

### (Optional) Run test suite
```bash
python -m unittest tests/test_api_predict.py
```

---

## **3. Workflow Overview**
<img width="2172" height="337" alt="Mermaid_Workflow_Overview" src="https://github.com/user-attachments/assets/97c375dc-6741-4270-bd9f-888d1e04a809" />

---

## **4. Model Training and Monitoring**

- `train_model_full_set.py` trains a ResNet50 classifier with learning rate scheduling and early stopping.  
- Accuracy and loss curves are logged and visualized.  
- The trained model (`refund_classifier_final.pt`) is excluded from the repository due to file size.  
- Full training logs and metrics are saved in:  
  `refund_classifier_training_log.pdf`

### Training Progress Output

The following terminal output shows real-time progress of training the ResNet50 model over 18 epochs:

<img width="1700" height="887" alt="training_progress_api_accuracy_loss" src="https://github.com/user-attachments/assets/04da2685-2a8d-4fb4-98e1-ef6ecb379286" />


### Training Accuracy and Loss Curve

Generated by `visualize_training_results.py`

![Training Curve](model_images/training_curve_final_model.png)
---

## **5. Prediction Interfaces**

- `app.py` exposes a REST API (`/predict`) that supports both single and batch image prediction.
- `batch_predict.py` performs automated classification on new images in batch mode (e.g., nightly schedule).
- Results are logged in PostgreSQL including filename, predicted class, and timestamp.

---

## **6. Visualization**

- `visualize_training_results.py`: Plots training accuracy and loss curves
- `confusion_matrix_heatmap.py`: Creates a normalized confusion matrix for validation data

### Confusion Matrix (Validation)
Generated by `confusion_matrix_heatmap.py`

![Confusion Matrix](model_images/confusion_matrix_validation_normalized.png)

Visualizations are also stored in `model_images/` and included in the PDF log.

---

## **7. Containerization and Testing**

- `docker-compose.yml`: Runs the API and database in containers  
- `Dockerfile`: Builds the REST API service  
- `.env`: Used for environment variables (excluded from GitHub)  
- `tests/test_api_predict.py`: Unit test to validate the prediction endpoint

### Docker Image Backup

The Docker image used in this project has been successfully saved as a `.tar` archive for reproducibility and redeployment:

```text
File: image_refund_classifier_api.tar
Location: D:\image_refund_classifier
Size: ~12 GB (complete Docker image)
```

<img width="773" height="607" alt="docker_container_startup_log" src="https://github.com/user-attachments/assets/458f4b06-66e3-4e03-bbd9-65044b30722a" />


---

## **8. Documentation**

- All training metrics, performance graphs, and implementation notes are available in:  
  `refund_classifier_training_log.pdf`
- All Python scripts include header docstrings describing their function and usage.

---

## **9. Project Structure**

```
DLBDSMTP01-From-Model-to-Production/
├── app/                           # Utility functions and model logic used by the API
│ ├── init.py
│ ├── model.py                     # Loads the trained model
│ ├── predict.py                   # Handles prediction logic
│ └── utils.py                     # Helper functions (e.g. image preprocessing)
├── data/
│ └── full/                        # Preprocessed dataset folder (train/val + styles.csv)
│ ├── train/                       # Training images organized by class
│ ├── val/                         # Validation images organized by class
│ └── styles.csv                   # Metadata from Kaggle dataset
├── model_images/                  # All generated plots and screenshots for documentation
│ ├── confusion_matrix_validation_normalized.png
│ ├── training_curve_final_model.png
│ ├── training_progress_api_accuracy.png
│ ├── training_terminal_output.png
│ └── docker_container_startup_log.png
├── tests/
│ └── test_api_predict.py          # Unit test for API predictions
├── .gitignore                     # Files and folders to ignore in Git version control
├── Dockerfile                     # Docker build instructions for the API
├── docker-compose.yml             # Defines and runs API + PostgreSQL as services
├── .env.example                   # Template for required environment variables
├── README.md                      # Project overview and instructions (this file)
├── app.py                         # Flask REST API entry point
├── batch_predict.py               # Batch prediction runner for incoming images
├── check_data.py                  # Verifies data structure and image paths
├── clean_class_folders.py         # Removes empty or unused folders (optional maintenance)
├── confusion_matrix_heatmap.py    # Creates evaluation heatmap of validation predictions
├── make_full_set.py               # (optional) Prepares filtered subset from raw Kaggle images
├── organize_by_class.py           # Moves images into class-labeled subfolders
├── prepare_full_dataset.py        # Splits images into train/val folders
├── refund_classifier_training_log.pdf # Final report with plots and performance notes
├── refund_classifier_final.pt     # Saved PyTorch model file
├── requirements.txt               # Python dependencies for the project
├── run_pipeline.sh                # Shell script to run the full project pipeline
├── sync_class_folders.py          # Ensures train/val contain the same classes
├── train_model_full_set.py        # Trains the image classifier and logs to MLflow
└── visualize_training_results.py  # Generates training accuracy and loss plots
---

## **10. Notes**

- Make sure to manually download and place the dataset in the correct folder paths before running any scripts.  
- Adjust paths in scripts if needed for your environment.  
- All components are included to reproduce the pipeline, excluding the dataset and final model weights.  
- Visuals and performance reports can be found in the attached PDF log.

