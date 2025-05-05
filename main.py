from loguru import logger
from pathlib import Path
import yaml
import typer
from typing import Dict, Any
from dvc.api import params_show
from src.dataset import process_data
from src.features import create_features
from src.plots import create_plots
from src.modeling.train import fetch_data, train, save_model
from src.modeling.predict import evaluate

def main():
    cfg = params_show()
    
    logger.info("Pipeline Started with DVC integration............")
    logger.info(f"Pipeline Parameters:\n{yaml.dump(cfg, indent=2)}")
    
    create_features()
    create_plots()
    process_data()
    x_train, y_train, x_test, y_test = fetch_data()
    model = train(x_train, y_train)
    save_model(model)
    evaluate(x_test, y_test)
    
    logger.success("Pipeline finished successfully with DVC tracking")

main()