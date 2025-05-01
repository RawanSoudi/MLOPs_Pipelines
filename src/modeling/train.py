from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import os
import pickle
import pandas as pd
from omegaconf import DictConfig
from  src.lr.lr import LR_Estimator
from src.dt.dt import DT_Estimator


Model_maps = {
    "lr": LR_Estimator,
    "dt": DT_Estimator
}

def fetch_data(cfg:DictConfig):
    SOURCE = cfg.model.processed_data_path
    MODELS_DIREC = cfg.model.model_path
    logger.info(f"Fetching Processed data from {SOURCE}...")
    files = {
        'X_train': 'processed_Xtrain.csv',
        'y_train': 'processed_ytrain.csv',
        'X_test': 'processed_Xtest.csv',
        'y_test': 'processed_ytest.csv'
    }
    
    data = {}
    for name, filename in files.items():
        filepath = os.path.join(cfg.data.processed_data_path, filename)
        data[name] = pd.read_csv(filepath)
    logger.success("Processed Data loaded successfully")
    logger.success(f"X_train shape {data['X_train'].shape}, Y_train shape {data['y_train'].shape}")
    logger.success(f"X_test shape {data['X_test'].shape}, Y_test shape {data['y_test'].shape}")
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']
   

def train(cfg:DictConfig, x_train,y_train):
    model =cfg.model_name
    logger.info(f"Training {model}...")
    model_class = Model_maps.get(model.lower())
    model = model_class()
    logger.info(f"Training {model} with {len(x_train)} samples...")
    model.fit(x_train, y_train.values.ravel())
    logger.success(f"{model} training completed")      
    return model

def save_model(model, cfg: DictConfig):
    os.makedirs(cfg.model_path, exist_ok=True)
    model_path = os.path.join(cfg.model_path, f"{cfg.model_name}_model.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.success(f"Model saved to {model_path}")

"""
def main(model_type="lr"):
    try:
        X_train, y_train, X_test, y_test = fetch_data(SOURCE)
        model = train(model_type, X_train, y_train)
        save_model(model, model_type)
        
        logger.success("Training pipeline completed successfully")
    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")
        raise typer.Exit(code=1)
"""