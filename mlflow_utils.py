import mlflow
import mlflow.sklearn
from mlflow.data import from_pandas
from typing import Dict, Optional

def set_tracking_uri(uri : Optional[str] = "http://127.0.0.1:5000/"):
    """
    Docstring for set_tracking_uri
    
    :param uri: Description
    """
    mlflow.set_tracking_uri(uri)

def set_experiment(experiment_name):
    """
    Create or set an MLflow experiment.

    :param experiment_name: Description
    """
    mlflow.set_experiment(experiment_name)

def start_run(nested : bool,run_name: Optional[str] = None):
    """
    Start an MLflow run.
    
    :param run_name: Description
    """
    mlflow.start_run(nested=nested,run_name=run_name)

def end_run ():
    """
    End the active MLflow run.
    
    :param end_run: Description
    """
    mlflow.end_run()

def log_params(params : dict):
    """
    Log hyperparameters.
    
    :param log_params: Description
    :type log_params: dict
    """
    mlflow.log_params(params)

def log_metrics (metrics:dict):
    """
    Log evaluation metrics.

    :param log_metrcies: Description
    :type log_metrcies: dict
    """
    mlflow.log_metrics(metrics)

def log_dataset(df,name,context):
    """
    Log a pandas DataFrame as an MLflow input dataset.

    context examples: 'train', 'validation', 'test'
    
    :param df: Description
    :param type: Description
    """
    dataset=from_pandas(df,name=name)
    mlflow.log_input(dataset,context=context)

def log_tags(tags:dict):
    """
    Set tags for the current MLflow run.
    
    :param tags: Description
    :type tags: dict
    """
    mlflow.set_tags(tags)

def log_artifacts(filepath:str):
    """
    Log artifacts (file or directory)
    
    :param filepath: Description
    :type filepath: str
    """
    mlflow.log_artifacts(filepath)

def log_sklearn_model(model, artifact_path:str):
    """
    Log a scikit-learn model to MLflow.
    
    :param model: Description
    :param artifact_path: Description
    :type artifact_path: str
    """
    mlflow.sklearn.log_model(sk_model=model,artifact_path=artifact_path)
