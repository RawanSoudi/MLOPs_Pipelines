from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import os
from omegaconf import DictConfig



def create_features(cfg : DictConfig):
    SOURCE = cfg.raw_data_path
    input_path = os.path.join(SOURCE, f"{cfg.file_name}")
    feature_path = cfg.feature_path
    
    logger.info(f"loading data from {input_path}...")

    try:
        df = pd.read_csv(input_path)
        df.columns = [x.lower() for x in df.columns]
        logger.success(f"Data loaded successfully (shape: {df.shape})")
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
        raise typer.Exit(code=1)

    logger.info(f"Feature Engineering...")
    try:
        df['family_size'] = df['sibsp'] + df['parch']
        df['is_alone'] = df['family_size'].apply(lambda x: 1 if x == 0 else 0)
        df.loc[(df['age'] >= 0)  & (df['age'] < 16),  'age_group'] = 'Below 16'
        df.loc[(df['age'] >= 16)  & (df['age'] < 40),  'age_group'] = 'Between 16-40'
        df.loc[(df['age'] >= 40)  & (df['age'] < 60),  'age_group'] = 'Between 40-60'
        df.loc[(df['age'] >= 60),  'age_group'] = 'Above 60'
        df.to_csv(feature_path, index=False)
        logger.success(f"Feature Engineering done successfully (shape: {df.shape})")
        logger.success(f"Features saved to {feature_path}")
    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        raise typer.Exit(code=1)


