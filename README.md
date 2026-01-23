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

### End-to-End Validation (PowerShell)

This command runs the complete local workflow:
- executes tests
- builds the Docker image
- starts the API container
- checks health endpoint
- sends a prediction request
- cleans up the container

> ⚡ **One-command end-to-end validation (PowerShell)**
>
> Paste and run from the project root. This will test, build, run, validate, and clean up the API.

```powershell
$ErrorActionPreference="Stop";

pytest -v;

docker build -t wine-api:latest .;

docker run -d --name wine_api -p 8000:8000 wine-api:latest;

Start-Sleep -Seconds 5;

Invoke-RestMethod http://127.0.0.1:8000/health;

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -InFile "payload.json";

docker stop wine_api;

docker rm wine_api;
```

**Requirements**
- Docker running
- Python + pytest installed
- PowerShell (Windows) or compatible shell


### 2. Debug / developer mode

- Run `train.py` locally
- Start the API using Uvicorn
- Validate predictions via browser or HTTP requests
- Execute pytest to verify API behavior

> 🐛 **Debug mode: build image, run API, inspect behavior**
>
> Intended for local debugging. Runs the container in the foreground so logs are visible.

```powershell
$ErrorActionPreference="Stop";

# 1️⃣ Install dependencies (if not already installed)
pip install -r requirements.txt;

# 2️⃣ Run training & evaluation locally
python train.py;

# 3️⃣ Start the API using Uvicorn (debug mode with hot reload)
uvicorn api:app --host 127.0.0.1 --port 8000 --reload;

# 4️⃣ Interact with the API via browser (GUI)
# Open the following URLs:
# - Swagger UI (interactive GUI for predictions):
http://127.0.0.1:8000/docs
# - Health check:
http://127.0.0.1:8000/health

# 5️⃣ Validate predictions via HTTP (run in another terminal)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -InFile "payload.json"

# 6️⃣ Execute pytest to verify API behavior
 pytest -v

```

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

