from pathlib import Path
from loguru import logger
import typer
import json
import os
import pickle
import pandas as pd
import yaml
from typing import Dict, Any 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from time import time



def evaluate(X_test, y_test,cfg: Dict[str, Any]):
    MODEL_PATH = cfg["evaluate"]["model_path"]
    REPORT_PATH = cfg["evaluate"]["reports_path"]
    logger.info(f"loading model {cfg["evaluate"]["model_name"]}.......")
    with open(os.path.join(MODEL_PATH, f"{cfg["evaluate"]["model_name"]}_model.pkl"), "rb") as pkl:
        model = pickle.load(pkl)
    start_time = time()
    y_pred = model.predict(X_test)
    prediction_time = time() - start_time
    logger.info("Calculating evaluation metrics...")
    metrics = {
            "model_name": cfg["evaluate"]["model_name"],
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "prediction_time_sec": prediction_time,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
    logger.info("Saving Evaluation Report...")
    os.makedirs(os.path.join(REPORT_PATH, cfg["evaluate"]["model_name"]), exist_ok=True)
    report_path = os.path.join(REPORT_PATH, cfg["evaluate"]["model_name"], "evaluation_report.json")
        
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=4)
            
    logger.success(f"Evaluation report saved to {report_path}")

