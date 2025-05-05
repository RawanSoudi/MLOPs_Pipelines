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
    
    logger.info("Pipeline Started with DVC............")
    logger.info(f"Pipeline Parameters:\n{yaml.dump(cfg, indent=2)}")
    
    create_features(cfg)
    create_plots(cfg)
    process_data(cfg)
    x_train, y_train, x_test, y_test = fetch_data(cfg)
    model = train(x_train, y_train,cfg)
    save_model(model,cfg)
    evaluate(x_test, y_test,cfg)
    
    logger.success("Pipeline finished successfully with DVC")

main()