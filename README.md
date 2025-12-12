# CPT Model Training, Prediction & Dashboard

This repository contains notebooks and configuration to:

1. Prepare CPT data from a single parquet file.
2. Train several machine-learning models (with and without coordinates).
3. Generate predictions and save them as a parquet file.
4. Explore results in an interactive Streamlit dashboard.

---

## Data and model distribution

All **data files and trained models** used by this project — including parquet files and model artifacts — are **not stored directly in this repository**.

Instead, they are downloaded from **Google Drive using `gdown`** inside the provided notebooks.

- Parquet data files are downloaded via the data download / preparation notebooks.
- Trained model files are downloaded or regenerated via the training notebooks.
- Once downloaded, files are saved locally and reused by the training, prediction,
  and dashboard workflows.

The **only exception** is the `Dashboard/` application code itself, which lives
entirely in this repository.

If required parquet or model files are missing, run the appropriate notebook to
download them before proceeding.


## Repository structure

- `Data_preperation.ipynb`  
  Splits the original input parquet into two parquet files:
  - one with rows that already have targets (used for training/validation),
  - one with rows that need predictions.  
  Filenames and paths are defined inside the notebook.

- `training_with_coordinations.ipynb`  
  Main training pipeline.  
  - Loads the “training” parquet from the preparation step.  
  - Trains all models using coordinate features.  
  - Saves trained models into the `trained_models/` folder.

- `training_No_coordinations.ipynb`  
  Alternative training pipeline without coordinate features.  
  It is **provided but not used by default**.  
  If you prefer models without coordinates, use this notebook instead of the one above.

- `prediction.ipynb`  
  - Loads the parquet with rows to be predicted.  
  - Loads all models from `trained_models/`.  
  - Produces a parquet file with model predictions.

- `trained_models/`  
  Contains the trained model files. You can use them directly for prediction without retraining.

## Dashboard

The dashboard is a **Streamlit application** located in the `Dashboard/` folder.
It is used to explore CPT data and model outputs interactively.

### Requirements
Before running the dashboard, make sure that:

- Your Python environment includes all required dependencies  
  (see `Dashboard/environment.yml` or the main environment configuration).
- You can run Streamlit (`streamlit` is installed).
- The required **parquet data files are available locally**.

### Data availability
The dashboard **does not download data automatically**.

If you do **not** already have the required parquet files:
1. Open `Files_downloader.ipynb` in the `Dashboard/` folder.
2. Run the notebook to download the necessary CPT parquet files
   (via Google Drive).
3. Verify that the downloaded parquet files are stored in the locations
   expected by `app.py`.

If the parquet files already exist, this step can be skipped.

### Running the dashboard
Once the environment and data are ready:

1. Open a terminal.
2. Navigate to the `Dashboard/` folder:
   ```bash
   streamlit run app.py

