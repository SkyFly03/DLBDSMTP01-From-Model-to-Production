# DLBDSMTP01 – From Model to Production  
## Task 2: Image Classification for a Refund Department (Batch Processing)

---

## **1. Objective**

The goal of this project is to build a batch-processing image classification system that automatically sorts returned products into predefined categories to support the refund department. This is achieved by:

- Training a deep learning model (ResNet-50) to classify product images into 31 refund categories.
- Preprocessing the dataset by cleaning missing images and filtering rare classes.
- Structuring the model pipeline into a batch-based system triggered by a scheduled job.
- Supporting reproducibility by training the model in both Google Colab (Model 1) and locally via shell script (Model 2).
- Integrating a REST API (using Flask and Docker) for scalable serving and batch prediction.
- Logging batch results to PostgreSQL for traceability and system monitoring.

---

## **2. Setup and Installation**

* ### Clone the repository
```bash
git clone https://github.com/SkyFly03/DLBDSMTP01-From-Model-to-Production.git
cd DLBDSMTP01-From-Model-to-Production
```

* ### Create and activate a virtual environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

* ### Install dependencies
```bash
pip install -r requirements.txt
```

* ### Download dataset manually from Kaggle
- Source: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset  
- Place `styles.csv` in `data/full/`  

Final folder structure after processing:
```
data/full/train/     # Training images  
data/full/val/       # Validation images  
```

* ### Prepare dataset
Use the scripts below to process and organize the dataset before training:
- `make_full_set.py`, `prepare_full_dataset.py`, `organize_by_class.py`, `sync_class_folders.py`

* ### Start PostgreSQL and REST API
```bash
docker-compose up -d
```

* ### (Optional) Run test suite
```bash
python -m unittest tests/test_api_predict.py
```
---

## **System Workflow – Batch Processing Pipeline**
<img width="2172" height="337" alt="Mermaid_Workflow_Overview" src="https://github.com/user-attachments/assets/97c375dc-6741-4270-bd9f-888d1e04a809" />

* Images are downloaded, cleaned, structured, and used to train a ResNet-50 model.
* The trained model is deployed via a REST API and batch predictions are logged to PostgreSQL.

## **3. Data Cleaning and Preparation**

Before training the models, the raw image dataset was processed using the following steps:

- `make_full_set.py`: Filters and copies a subset of product images from the full Kaggle dataset based on selected categories.
- `prepare_full_dataset.py`: Randomly splits the dataset into training and validation sets with equal distribution.
- `organize_by_class.py`: Sorts images into class-labeled subdirectories required for PyTorch's `ImageFolder` format.
- `sync_class_folders.py`: Ensures both `train/` and `val/` folders contain the exact same set of class directories.
- `check_data.py`: Verifies that all images are accessible and folders are correctly organized.
- Duplicate or corrupted images are skipped automatically during loading via built-in checks.

---

## **4. Model Training and Monitoring**

- A ResNet-50 model was trained using two setups: Colab (GPU) and local shell script (CPU) for reproducibility.
- Training used learning rate scheduling and early stopping.
- Models are stored as `.pt` files (excluded from GitHub).
- **Architecture**: Pretrained ResNet-50 with a custom output layer for 31 refund classes.

### **4.1 Model 1 – Google Colab (GPU)**

- Trained for 18 epochs using Colab Pro GPU runtime

### Training Output:
<img width="1700" height="887" alt="training_progress_Colab" src="https://github.com/user-attachments/assets/014e5c88-157e-4d90-8472-d4dc997c7acb" />

### Training Accuracy and Loss Curve:
<img width="988" height="390" alt="training_curve_model1_colab" src="https://github.com/user-attachments/assets/5288b93e-2191-4828-90c9-92e0470f38a9" />

### Confusion Matrix (Validation – Colab):
<img width="1129" height="989" alt="confusion_matrix_colab_model" src="https://github.com/user-attachments/assets/983751c0-dc10-4a66-b250-fd5f73e51c1a" />

---

### **4.2 Model 2 - Local Shell Script (Docker CPU)**

- Trained for 26 epochs via `run_pipeline.sh` using a containerized CPU environment

### Training Output:
<img width="344" height="341" alt="training_progress_shell2" src="https://github.com/user-attachments/assets/ceb17613-f369-4575-84bd-7d39f86e96ce" />

### Training Accuracy and Loss Curve:
<img width="909" height="403" alt="training_curve_model2_shell" src="https://github.com/user-attachments/assets/2be24297-b810-4651-96e4-815119de0abd" />

### Confusion Matrix (Validation – Shell):
<img width="1200" height="1000" alt="confusion_matrix_shell_model" src="https://github.com/user-attachments/assets/a8991b53-dc53-48b7-abf5-8dbeb465cb43" />

---

### **4.3 Comparison and Evaluation**

|                      | Model 1 – Colab        | Model 2 – Shell Script   |
|----------------------|------------------------|--------------------------|
| Train Accuracy       | 0.94                   | 0.977                    |
| Validation Accuracy  | 0.85                   | 0.86                     |
| Runtime              | ~1.5 hours (GPU)       | ~6.5 hours (CPU/Docker)  |
| Deployment Use Case  | Fast prototyping       | Stable batch deployment  |

- Both models achieved stable validation accuracy over 85%, fulfilling the project’s objective.
- Model 1 is ideal for rapid development and iteration using GPU resources.
- Model 2 is fully integrated in the system pipeline and better suited for long-term automated batch predictions (e.g., cronjob or nightly runs).

---

## **5. Prediction Interfaces**

- `app.py` exposes a REST API (`/predict`) that supports both single and batch image prediction.
- `batch_predict.py` performs automated classification on new images in batch mode (e.g., nightly schedule).
- **Prediction results** (filename, predicted class, and timestamp) are logged into a PostgreSQL database via SQLAlchemy for traceability and future monitoring.

---

## **6. Containerization and Testing**

- `docker-compose.yml`: Runs the API and database in containers  
- `Dockerfile`: Builds the REST API service  
- `.env`: Used for environment variables (excluded from GitHub)  
- `tests/test_api_predict.py`: Unit test to validate the prediction endpoint


### Docker Image Backup

A full Docker image backup (`image_refund_classifier_api.tar`) has been saved locally for reproducibility.

```text
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
│ ├── confusion_matrix_colab_model.png
│ ├── confusion_matrix_shell_model.png
│ ├── training_curve_model1_colab.png
│ ├── training_curve_model2_shell.png
│ ├── training_progress_colab.png
│ ├── training_progress_shell1.png
│ └── training_progress_shell2.png
├── notebooks/
│ └── image_refund_classifier_metadata.ipynb    
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
├── confusion_matrix_heatmap1_colab.py # Creates evaluation heatmap of validation predictions - Colab
├── confusion_matrix_heatmap2_shell.py # Creates evaluation heatmap of validation predictions - Shell
├── make_full_set.py               # (optional) Prepares filtered subset from raw Kaggle images
├── organize_by_class.py           # Moves images into class-labeled subfolders
├── prepare_full_dataset.py        # Splits images into train/val folders
├── refund_classifier_training_log.pdf # Final report with plots and performance notes
├── refund_classifier_final.pt     # Saved PyTorch model file
├── requirements.txt               # Python dependencies for the project
├── run_pipeline.ps1               # PowerShell script to automate the full local training and evaluation pipeline (Windows)
├── run_pipeline.sh                # Shell script to run the full project pipeline
├── sync_class_folders.py          # Ensures train/val contain the same classes
├── train_model_full_set.py        # Trains the image classifier and logs to MLflow
└── visualize_training_results.py  # Generates training accuracy and loss plots
```
---
