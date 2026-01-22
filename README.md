# ml-wine-quality-api

End-to-end ML project that trains a wine quality classifier and serves predictions via a FastAPI service, containerized with Docker.

---

## Overview

This project implements an end-to-end machine learning workflow with a strong focus on **deployment and reproducibility** rather than model complexity.

The task is to predict whether a wine is of *good quality* based on physicochemical features.  
The dataset is programmatically downloaded from the UCI Machine Learning Repository.  
Exploratory data analysis and model comparison were performed separately in Jupyter notebooks. Based on these experiments, a Logistic Regression model was selected and implemented in the production training pipeline.

The trained model is exposed via a FastAPI inference service and packaged using Docker.  
Deployment is validated using automated tests (`pytest`) and local Docker execution.

---

## Usage modes

This repository can be used in two modes:

1. **End-to-end user mode**
   - Run the complete system using Docker
   - Training, model loading, and API execution are handled inside the container 

2. **Debug / developer mode**
   - Run training and evaluation scripts manually  
   - Execute the API locally with FastAPI  
   - Validate endpoints using pytest  

---

## Design decisions

- Converted the original multi-class quality score into a binary target (`quality >= 7`) to simplify inference logic.
- A decision threshold of **0.7** is used for classifying good vs bad wine.
- Exploratory analysis indicated minimal preprocessing requirements.
- Logistic Regression was selected as a stable and interpretable baseline model.
- A stratified train/validation split is used to handle class imbalance.
- API endpoints are validated using pytest (3 test cases).
- Docker is used to ensure reproducible execution across environments.

---

## Data handling

- The dataset is downloaded directly from the UCI repository during training.
- Data is stored locally to ensure reproducibility and offline execution.
- Labels are created during training and are not persisted as part of the raw dataset.
- Stratified splitting is applied to preserve class distribution between train and validation sets.

---

## Project structure

- `train.py`: Orchestrates data loading, training, evaluation, and artifact saving  
- `api.py`: FastAPI application defining inference endpoints  
- `docker/`: Container configuration  
- `src/`: Core training and evaluation logic  
- `notebooks/`: Exploratory analysis and model experiments (not part of production pipeline)

---

## Training workflow

1. Download and persist dataset
2. Create binary target variable
3. Perform stratified train/validation split
4. Train the selected model
5. Evaluate performance on validation data
6. Save trained model artifact for inference

---

## Running the project

### 1. End-to-end mode (Docker)

- Build the Docker image
- Run the container
- Send a prediction request to the API endpoint

### 2. Debug / developer mode

- Run `train.py` locally
- Start the API using Uvicorn
- Validate predictions via browser or HTTP requests
- Execute pytest to verify API behavior

---

## Limitations

- No cloud deployment (local Docker only)
- Several parameters are hardcoded and could be externalized via configuration or argparse
- Model retraining is manual
- No monitoring or logging for deployed predictions

---

## Future improvements

- Add configurable training parameters
- Introduce automated retraining and CI/CD
- Add experiment tracking
- Deploy to a cloud environment
- Improve validation and monitoring for inference requests

