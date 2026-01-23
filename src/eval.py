# src/eval.py

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def predict_proba(pipe, X):
    # returns probability of class 1
    return pipe.predict_proba(X)[:, 1]


def predict_with_threshold(y_proba, threshold):
    return (y_proba >= threshold).astype(int)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def evaluate(pipe, X, y, threshold=0.7, verbose=False):
    y_proba = predict_proba(pipe, X)
    y_pred = predict_with_threshold(y_proba, threshold)

    results = {
        "threshold": threshold,
        "metrics": compute_metrics(y, y_pred, y_proba),
        "confusion_matrix": confusion_matrix(y, y_pred),
    }

    if verbose:
        results["classification_report"] = classification_report(y, y_pred)

    return results
