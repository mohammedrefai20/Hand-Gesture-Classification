# Hand Gesture Classification (Research Branch)

## 1) Objective
This repository implements a classical machine learning pipeline for **hand gesture classification** using hand-landmark coordinates, with complete experiment tracking in MLflow.

The workflow covers:
- Data preprocessing (`X`, `y` generation from a CSV file).
- Training and tuning of multiple models (SVM, KNN, Random Forest, AdaBoost).
- Experiment/run logging (dataset, parameters, metrics, model, artifacts).
- Model comparison and final model registration in MLflow Model Registry.

---

## 2) Project Structure

```text
.
├── mlflow_utils.py      # Centralized MLflow helper functions (separate script)
├── preprocessing.py     # CSV preprocessing and normalization -> X, y
├── svm_train.py         # SVM + GridSearchCV + MLflow logging
├── knn_train.py         # KNN + GridSearchCV + MLflow logging
├── rf_train.py          # Random Forest + GridSearchCV + MLflow logging
├── adb_train.py         # AdaBoost + GridSearchCV + MLflow logging
└── mlartifacts/         # Exported MLflow artifacts/model descriptors
```

---

## 3) MLflow Utility Module (Required Separate Script)

All reusable MLflow operations are centralized in `mlflow_utils.py`:

### Tracking & Experiment Management
- `set_tracking_uri(uri="http://127.0.0.1:5000/")`
- `set_experiment(experiment_name)`

### Run Lifecycle
- `start_run(run_name=None, nested=False)`
- `end_run()`

### Logging APIs
- `log_dataset(df, name, context)`
- `log_params(params)`
- `log_metrics(metrics)`
- `log_tags(tags)`
- `log_sklearn_model(model, artifact_path)`
- `log_artifacts(filepath)`

This satisfies the requirement to keep MLflow logic in a **separate Python file** and reuse it across training scripts.

---

## 4) Preprocessing

`preprocessing.py` exposes:

- `preprcessing(filepath: str) -> (X, y)`

### Processing steps
1. Read CSV input.
2. Translate landmarks by subtracting base point (`x1`, `y1`) from points `2..21`.
3. Compute normalization scale: `sqrt(x13^2 + y13^2)`.
4. Normalize all coordinates (`x1..x21`, `y1..y21`) by that scale.
5. Return:
   - `X`: normalized landmark features.
   - `y`: label column.

---

## 5) Training, Logging, and Experiment Design

Training and logging occur in these scripts:
- `svm_train.py`
- `knn_train.py`
- `rf_train.py`
- `adb_train.py`

Each script performs:
1. Load and preprocess the dataset.
2. Set MLflow tracking URI and experiment.
3. Split train/test (`test_size=0.2`, `random_state=42`).
4. Start a **parent run** (per model family).
5. Log datasets (`train`, `test`).
6. Run `GridSearchCV` (`cv=3`, `scoring='f1_weighted'`, `n_jobs=-1`).
7. Log each hyperparameter combination as a **nested run**.
8. Log best params and best model artifact.
9. Log test weighted F1 score.
10. Log confusion-matrix artifact chart(s).

### Experiment and run names used
- Experiment: `ML1_project`
- Parent runs:
  - `SVM_GridSearch`
  - `KNN_GridSearch`
  - `RandomForest_GridSearch`
  - `AdaBoost_GridSearch`

---

## 6) Hyperparameter Search Space

### SVM
- `kernel`: `['rbf']`
- `C`: `[100, 130, 150]`
- `gamma`: `[0.01, 0.05, 0.1]`

### KNN
- `n_neighbors`: `[3, 5, 7, 9, 11]`
- `weights`: `['uniform', 'distance']`
- `metric`: `['euclidean', 'manhattan']`

### Random Forest
- `n_estimators`: `[100, 300, 400, 500, 600]`
- `max_depth`: `[5, 7, 9, 11]`

### AdaBoost
- `estimator__max_depth`: `[3, 5, 7, 11]`
- `n_estimators`: `[400, 500, 600]`
- `learning_rate`: `[0.5, 0.7, 0.8, 0.9]`

---

## 7) Model Comparison (Decision Support)

### Comparison table (weighted F1 on test set)

| Model | f1_score_test |
|---|---:|
| SVM (GridSearch) | 0.98 |
| KNN (GridSearch) | 0.95 |
| Random Forest (GridSearch) | 0.94 |
| AdaBoost (GridSearch) | 0.98 |

### Selection rationale
AdaBoost and SVM both achieved the highest reported test weighted F1 (`0.98`). The final registered model is **AdaBoostModel (Version 1)**, selected as the production candidate based on the experiment decision.

---

## 8) Representative Charts for Model Comparison

The following artifact charts are logged and used for model comparison and error analysis:

- SVM confusion matrices:  
  `mlartifacts/3/936b99b319fe449c902f4f4d600210c7/artifacts/svm_confusion_matrices.png`
- KNN confusion matrices:  
  `mlartifacts/3/936b99b319fe449c902f4f4d600210c7/artifacts/knn_confusion_matrices.png`
- Random Forest confusion matrices:  
  `mlartifacts/3/936b99b319fe449c902f4f4d600210c7/artifacts/rf_confusion_matrices.png`
- AdaBoost confusion matrices:  
  `mlartifacts/3/936b99b319fe449c902f4f4d600210c7/artifacts/adaboost_confusion_matrices.png`

These charts complement the metric table and support the final model selection.

---

## 9) How to Run

1. Start MLflow tracking server.
2. Update dataset path in each training script (currently hardcoded absolute path).
3. Run training scripts:

```bash
python svm_train.py
python knn_train.py
python rf_train.py
python adb_train.py
```

---

## 10) Model Registry

Registered model details:
- Model name: `AdaBoostModel`
- Version: `1`
- Selected model family: AdaBoost

