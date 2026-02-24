# Hand Gesture Classification with MLflow Tracking

## 1. Project Overview
This repository contains a complete classical machine learning workflow for hand gesture classification using landmark coordinates.

The pipeline includes:
- Centralized MLflow utility functions for experiment management, dataset/model logging, and run organization.
- A preprocessing module that normalizes hand landmarks and returns feature matrix `X` and labels `y`.
- Four model training workflows:
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - AdaBoost
- Hyperparameter optimization using `GridSearchCV` with weighted F1 score as the objective.
- Full experiment tracking in MLflow (parameters, metrics, artifacts, and logged models).

## 2. Repository Structure

```text
.
├── mlflow_utils.py      # Reusable MLflow helper functions
├── preprocessing.py     # Data loading + normalization -> X, y
├── svm_train.py         # SVM training + grid search + MLflow logging
├── knn_train.py         # KNN training + grid search + MLflow logging
├── rf_train.py          # Random Forest training + grid search + MLflow logging
├── adb_train.py         # AdaBoost training + grid search + MLflow logging
└── mlartifacts/         # Exported MLflow artifacts and model descriptors
```

## 3. MLflow Utilities (`mlflow_utils.py`)

The `mlflow_utils.py` file provides reusable wrappers around common MLflow operations.

### 3.1 Tracking and Experiment Control
- `set_tracking_uri(uri="http://127.0.0.1:5000/")`
  - Sets the MLflow tracking server URI.
- `set_experiment(experiment_name)`
  - Creates/selects the target MLflow experiment.

### 3.2 Run Lifecycle
- `start_run(run_name=None, nested=False)`
  - Starts a run (supports nested child runs for grid search combinations).
- `end_run()`
  - Ends the active MLflow run.

### 3.3 Logging Helpers
- `log_params(params: dict)`
  - Logs model hyperparameters.
- `log_metrics(metrics: dict)`
  - Logs evaluation metrics.
- `log_dataset(df, name, context)`
  - Logs input datasets to MLflow with context (e.g., `train`, `test`).
- `log_tags(tags: dict)`
  - Adds tags to the active run.
- `log_artifacts(filepath: str)`
  - Logs artifact files/directories.
- `log_sklearn_model(model, artifact_path: str)`
  - Logs a scikit-learn model using the MLflow sklearn flavor.

## 4. Preprocessing (`preprocessing.py`)

The preprocessing function is:

- `preprcessing(filepath: str) -> (X, y)`

### 4.1 What it does
1. Reads hand landmarks from CSV.
2. Translates all points by subtracting the base point `(x1, y1)` from landmarks `2..21`.
3. Computes scale using `sqrt(x13^2 + y13^2)`.
4. Normalizes all `x1..x21` and `y1..y21` coordinates by this scale.
5. Returns:
   - `X`: all normalized landmark features.
   - `y`: `label` column.

This makes the representation more robust to translation and scale variation.

## 5. Training and Experiment Workflow

Each training script follows the same structure:
1. Load and preprocess dataset using `preprcessing(...)`.
2. Configure MLflow (`set_tracking_uri`, `set_experiment`).
3. Split data into train/test (80/20, `random_state=42`).
4. Start parent run (model-level run).
5. Log train and test datasets.
6. Run `GridSearchCV` (`cv=3`, `scoring="f1_weighted"`, `n_jobs=-1`).
7. For each hyperparameter combination:
   - Start nested run.
   - Log parameters + mean CV and train F1.
   - End nested run.
8. Log best parameters and best model artifact.
9. Evaluate best model on test split and log `f1_score_test`.
10. Save and log confusion matrix analysis figure.

## 6. Hyperparameter Search Spaces

### 6.1 SVM (`svm_train.py`)
- `kernel`: `['rbf']`
- `C`: `[100, 130, 150]`
- `gamma`: `[0.01, 0.05, 0.1]`

### 6.2 KNN (`knn_train.py`)
- `n_neighbors`: `[3, 5, 7, 9, 11]`
- `weights`: `['uniform', 'distance']`
- `metric`: `['euclidean', 'manhattan']`

### 6.3 Random Forest (`rf_train.py`)
- `n_estimators`: `[100, 300, 400, 500, 600]`
- `max_depth`: `[5, 7, 9, 11]`

### 6.4 AdaBoost (`adb_train.py`)
- `estimator__max_depth`: `[3, 5, 7, 11]`
- `n_estimators`: `[400, 500, 600]`
- `learning_rate`: `[0.5, 0.7, 0.8, 0.9]`

## 7. Model Comparison (Test F1)

The tracked `f1_score_test` values from the experiment comparison are:

| Model | Test Weighted F1 |
|---|---:|
| SVM (GridSearch) | 0.98 |
| KNN (GridSearch) | 0.95 |
| Random Forest (GridSearch) | 0.94 |
| AdaBoost (GridSearch) | 0.98 |

The best performing configuration is registered as **AdaBoost** and assigned alias **`ada_boost_model`**.

## 8. Confusion Matrix Artifacts

The experiment logs confusion matrix analysis images for each model (counts, normalized matrix, and misclassified-only matrix):

### SVM
![SVM confusion matrices](mlartifacts/3/936b99b319fe449c902f4f4d600210c7/artifacts/svm_confusion_matrices.png)

### KNN
![KNN confusion matrices](mlartifacts/3/936b99b319fe449c902f4f4d600210c7/artifacts/knn_confusion_matrices.png)

### Random Forest
![Random Forest confusion matrices](mlartifacts/3/936b99b319fe449c902f4f4d600210c7/artifacts/rf_confusion_matrices.png)

### AdaBoost
![AdaBoost confusion matrices](mlartifacts/3/936b99b319fe449c902f4f4d600210c7/artifacts/adaboost_confusion_matrices.png)

## 9. How to Run

1. Start MLflow tracking server (or point to an existing URI).
2. Update dataset path in each training script (currently a local absolute path).
3. Run any training script:

```bash
python svm_train.py
python knn_train.py
python rf_train.py
python adb_train.py
```

## 10. Notes
- Current scripts use a hardcoded dataset path. For production usage, pass the dataset path as a CLI argument or environment variable.
- The preprocessing function name in code is `preprcessing` (spelling preserved from source implementation).
- MLflow model descriptors are present in `mlartifacts/3/models`, while large serialized model binaries are intentionally not included.
