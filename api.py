# api.py

import pandas as pd
from fastapi import FastAPI, HTTPException
from src.artifact import load_artifact

MODEL_PATH = "model.joblib"

app = FastAPI()

# Load once at startup
artifact = load_artifact(MODEL_PATH)
pipe = artifact["model"]
FEATURES = artifact["features"]
THRESHOLD = artifact["threshold"]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "threshold": THRESHOLD,
        "n_features": len(FEATURES),
    }


@app.post("/predict")
def predict(payload: dict):
    # payload is a dict of feature -> value
    X_new = pd.DataFrame([payload]).reindex(columns=FEATURES)

    # missing check
    if X_new.isna().any().any():
        missing = X_new.columns[X_new.isna().any()].tolist()
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features: {missing}. Expected: {FEATURES}"
        )

    prob = float(pipe.predict_proba(X_new)[:, 1][0])
    pred = int(prob >= THRESHOLD)

    return {
        "good_quality": pred,
        "probability": prob,
        "threshold": THRESHOLD
    }

### Example API Response
'''
Request: Example for API:
```json 
{
  "fixed acidity": 7.4,
  "volatile acidity": 0.7,
  "citric acid": 0.0,
  "residual sugar": 1.9,
  "chlorides": 0.076,
  "free sulfur dioxide": 11,
  "total sulfur dioxide": 34,
  "density": 0.9978,
  "ph": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
} 
expected output:
{
  "good_quality": 0,
  "probability": 0.03517747236025412,
  "threshold": 0.7
}

################# Another Example Request ###################
Request:
{
  "fixed acidity": 8.3,
  "volatile acidity": 0.3,
  "citric acid": 0.4,
  "residual sugar": 1.8,
  "chlorides": 0.06,
  "free sulfur dioxide": 15,
  "total sulfur dioxide": 50,
  "density": 0.995,
  "ph": 3.3,
  "sulphates": 0.8,
  "alcohol": 12.5
}
expected output:
{
  "good_quality": 1,
  "probability": 0.9141792343032167,
  "threshold": 0.7
}
'''