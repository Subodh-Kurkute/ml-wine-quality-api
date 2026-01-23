# Source Code (`src/`)

This directory contains the core implementation of the wine quality prediction system.  
The code is structured to separate **data handling**, **model logic**, **evaluation**, and **artifacts**, following production-style ML project conventions.

---

## Module Overview

### `data_loader.py`
Handles dataset access and preparation.

Responsibilities:
- Download the Wine Quality dataset from the UCI repository
- Load the dataset into memory
- Perform minimal preprocessing (no model logic)

---

### `model.py`
Defines the machine learning model.

Responsibilities:
- Build and configure the classification model
- Expose a clean interface for training and inference
- Keep training loops outside this module

---

### `eval.py`
Contains evaluation utilities.

Responsibilities:
- Compute evaluation metrics (e.g. ROC-AUC, F1-score)
- Support threshold-based performance analysis
- Avoid data loading or training logic

---

### `artifact.py`
Manages model artifacts.

Responsibilities:
- Save trained models to disk
- Load models for inference
- Centralize filesystem and artifact handling logic

---

### `schema.py`
Defines input and feature schemas.

Responsibilities:
- Specify model feature names and order
- Validate request payload structure
- Ensure consistency between training and inference

---

### `__init__.py`
Marks the directory as a Python package and enables clean imports.

---

## Design Principles

- Single-responsibility modules
- Clear separation of training, evaluation, and serving
- Reproducible and CI-friendly structure
- No hard dependency on runtime environment (Docker / local)

---

## Notes

- No model artifacts are stored in this directory
- All generated files are excluded via `.gitignore`
- Modules are designed to be reusable across training, API, and CI workflows
