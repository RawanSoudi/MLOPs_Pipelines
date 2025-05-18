from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import os
import pickle
import pandas as pd
import yaml  
from typing import Dict, Any 
from  src.lr.lr import LR_Estimator
from src.dt.dt import DT_Estimator

from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV

import dvc.api
import mlflow
import mlflow.sklearn
import dagshub


load_dotenv()

Model_maps = {
    "lr": LR_Estimator,
    "dt": DT_Estimator
}

def fetch_data(cfg: Dict[str, Any]):
    SOURCE = cfg["model"]["processed_data_path"]
    MODELS_DIREC = cfg["model"]["model_path"]
    logger.info(f"Fetching Processed data from {SOURCE}...")
    files = {
        'X_train': 'processed_Xtrain.csv',
        'y_train': 'processed_ytrain.csv',
        'X_test': 'processed_Xtest.csv',
        'y_test': 'processed_ytest.csv'
    }
    
    data = {}
    for name, filename in files.items():
        filepath = os.path.join(cfg["data"]["processed_data_path"], filename)
        data[name] = pd.read_csv(filepath)
    logger.success("Processed Data loaded successfully")
    logger.success(f"X_train shape {data['X_train'].shape}, Y_train shape {data['y_train'].shape}")
    logger.success(f"X_test shape {data['X_test'].shape}, Y_test shape {data['y_test'].shape}")
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']
   

def train(x_train,y_train,cfg: Dict[str, Any]):
    params = {
    'min_samples_split': [2, 3, 4],
    'max_depth': [None, 5, 10]
}
    model =cfg["model"]["model_name"]
    logger.info(f"Training {model}...")
    model_class = Model_maps.get(model.lower())
    model = model_class()
    dagshub.auth.add_app_token(os.getenv("DAGSHUB_TOKEN"))

    dagshub.init(
        repo_name=os.getenv("DAGSHUB_REPO"),
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        mlflow=True
    )
    mlflow.set_experiment("Titanic-GridSearch")

    mlflow.sklearn.autolog()
    grid_search = GridSearchCV(model, params, cv=5)
    grid_search.fit(x_train, y_train.values.ravel())
    model = grid_search.best_estimator_

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="DT_Titanic"
    )
 
    logger.info(f"Training {model} with {len(x_train)} samples...")
    logger.success(f"{model} training completed")      
    return model

def save_model(model,cfg: Dict[str, Any]):
    os.makedirs(cfg["model"]["model_path"], exist_ok=True)
    model_path = os.path.join(cfg["model"]["model_path"], f"{cfg["model"]["model_name"]}_model.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.success(f"Model saved to {model_path}")

