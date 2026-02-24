# Imports
import pandas as pd
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mlflow_utils as mu
import preprocessing as p 

# =============================== 
# Load Dataset
# =============================== 
X, y = p.preprcessing(r'G:\ITI_AI\Lectures\Machine Learning\Project\Dataset\hand_landmarks_data.csv')

# =============================== 
# MLflow Setup
# =============================== 
mu.set_tracking_uri("http://127.0.0.1:5000/")
mu.set_experiment("ML1_project")

# =============================== 
# Train-Test Split
# =============================== 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================== 
# Start Parent Run
# =============================== 
mu.start_run(run_name="AdaBoost_GridSearch")

# Log datasets
mu.log_dataset(X_train, name="Training Dataset", context="train")
mu.log_dataset(X_test, name="Test Dataset", context="test")

# =============================== 
# Grid Search Setup
# =============================== 
param_grid = {
    'estimator__max_depth': [3,5,7,11],
    'n_estimators': [400,500,600],
    'learning_rate': [0.5,0.7,0.8,0.9]
}

grid_ada = GridSearchCV(
    estimator=AdaBoostClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        random_state=42
    ),
    param_grid=param_grid,
    scoring="f1_weighted",
    verbose=1,
    n_jobs=-1,
    cv=3,
    return_train_score=True
)

# Fit grid search
grid_ada.fit(X_train, y_train)

# =============================== 
# Log All Grid Search Results as Nested Runs
# =============================== 
for i, params in enumerate(grid_ada.cv_results_['params']):
    # Create descriptive run name
    run_name = f"AdaBoost/n_est={params['n_estimators']}/lr={params['learning_rate']}/depth={params['estimator__max_depth']}/run_No.{i}"
    
    # Start nested run for this parameter combination
    mu.start_run(run_name=run_name, nested=True)
    
    # Log hyperparameters
    mu.log_params(params)
    
    # Log all available metrics for this combination
    metrics = {
        "mean_cv_f1_score": grid_ada.cv_results_['mean_test_score'][i],
        "mean_train_score": grid_ada.cv_results_['mean_train_score'][i],
    }
    mu.log_metrics(metrics)
    
    # End nested run
    mu.end_run()

# =============================== 
# Log Best Model Info in Parent Run
# =============================== 
best_model = grid_ada.best_estimator_

# Log best parameters
mu.log_params(grid_ada.best_params_)

# Log best model
mu.log_sklearn_model(best_model, artifact_path="best_adaboost_model")

# =============================== 
# Evaluate Best Model on Test Set
# =============================== 
y_pred = best_model.predict(X_test)
f1 = f1_score(y_test, y_pred, average="weighted")
mu.log_metrics({"f1_score_test": f1})

# =============================== 
# Generate and Log Confusion Matrix
# =============================== 
# Create artifacts directory
artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

# Create figure with 3 confusion matrices
fig, axes = plt.subplots(3, 1, figsize=(30, 40))

# Confusion Matrix (Counts)
cm_counts = confusion_matrix(y_test, y_pred)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_counts, display_labels=best_model.classes_)
disp1.plot(ax=axes[0])
axes[0].set_title("Confusion Matrix (Counts)")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)

# Normalized Confusion Matrix
cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=best_model.classes_)
disp2.plot(ax=axes[1], values_format=".00%")
axes[1].set_title("Normalized Confusion Matrix")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)

# Misclassified Samples Only
sample_weight = (y_pred != y_test)
cm_error = confusion_matrix(y_test, y_pred, sample_weight=sample_weight, normalize="true")
disp3 = ConfusionMatrixDisplay(confusion_matrix=cm_error, display_labels=best_model.classes_)
disp3.plot(ax=axes[2], values_format=".00%")
axes[2].set_title("Misclassified Samples Only")
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=90)

fig.suptitle("AdaBoost Confusion Matrices Analysis", fontsize=16, y=0.98)
plt.savefig("artifacts/adaboost_confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.close()

# Log artifacts directory
mu.log_artifacts(artifacts_dir)

# =============================== 
# End Parent Run
# =============================== 
mu.end_run()

print(f"\n Grid Search Complete!")