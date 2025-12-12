# CPT Model Training, Prediction & Dashboard

This repository contains notebooks and configuration to:

1. Prepare CPT data from a single parquet file.
2. Train several machine-learning models (with and without coordinates).
3. Generate predictions and save them as a parquet file.
4. Explore results in an interactive Streamlit dashboard.

---

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

- `Dashboard/`  
  Streamlit dashboard for exploring results.  
  Run from inside this folder with:
  ```bash
  streamlit run app.py
