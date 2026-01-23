# src/artifact.py

import joblib

def save_artifact(pipe, features, threshold, path="model.joblib"):
    artifact = {
        "model": pipe,
        "features": features,
        "threshold": threshold,
    }
    joblib.dump(artifact, path)
    return path


def load_artifact(path="model.joblib"):
    artifact = joblib.load(path)
    return artifact
