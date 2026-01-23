import pandas as pd
from sklearn.model_selection import train_test_split

from src.schema import FEATURES
from src.model import build_model
from src.eval import evaluate
from src.artifact import save_artifact
from src.data_loader import load_red_wine_quality, download_red_wine_quality

LABEL_THRESHOLD = 7                # good_quality = (quality >= 7)
DECISION_THRESHOLD = 0.7           # model decision threshold used in deployment
RANDOM_STATE = 42

def main():
    # 1) Load data
    data_path = download_red_wine_quality()
    df = load_red_wine_quality(data_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    print(df.shape)

    # 2) Create binary label (training only)
    df["good_quality"] = (df["quality"] >= LABEL_THRESHOLD).astype(int)

    # 3) Build X, y
    X = df[FEATURES].copy()
    y = df["good_quality"].copy()

    # 4) Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # 5) Train model
    pipe = build_model()
    pipe.fit(X_train, y_train)

    # 6) Evaluate (train + val)
    print("\n=== TRAIN ===")
    train_res = evaluate(pipe, X_train, y_train, threshold=DECISION_THRESHOLD, verbose=True)
    print(train_res["metrics"])
    print(train_res["classification_report"])

    print("\n=== VALIDATION ===")
    val_res = evaluate(pipe, X_val, y_val, threshold=DECISION_THRESHOLD, verbose=True)
    print(val_res["metrics"])
    print(val_res["classification_report"])

    # 7) Save artifact for API
    save_path = save_artifact(pipe, FEATURES, DECISION_THRESHOLD, path="model.joblib")
    print(f"\nSaved model artifact to: {save_path}")

if __name__ == "__main__":
    main()