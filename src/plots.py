from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from omegaconf import DictConfig



def count_plot(col,orientation,data):
    plt.figure(figsize=(8,6))
    order = data[col].value_counts().index
    counts = data[col].value_counts()
    palette = ['steelblue' if category == counts.idxmax() else 'grey' for category in order]
    if orientation == 'vertical':
        ax = sns.countplot(data = data, x = col,order=order, palette=palette)
        ax.set(xlabel=None)
    else:
        ax = sns.countplot(data=data, y=col, order=order, palette=palette)
        ax.set(ylabel=None)
    column = col.replace('_',' ').title()
    plt.title(column+' Distribution')
    sns.despine()

    

def create_plots( cfg: DictConfig):
  
    DESTINATION = cfg.evaluate.figure_path
    input_path = cfg.data.feature_path
    logger.info(f"loading data from {input_path}...")

    try:
        df = pd.read_csv(input_path)
        df.columns = [x.lower() for x in df.columns]
        logger.success(f"Data loaded successfully (shape: {df.shape})")
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
        raise typer.Exit(code=1)

    logger.info(f"Creating plots...")
    sns.displot(data=df, x='age',kde=True,color='steelblue')
    plt.title('Age Distribution')
    output_path =  os.path.join(DESTINATION, "Age_Distribution.png")
    plt.savefig(output_path, dpi=300)
    logger.success(f"Age Distribution plot created successfully (Path: {output_path})")

    count_plot(col='survived',orientation='vertical',data=df)
    output_path =  os.path.join(DESTINATION, "survived_Distribution.png")
    plt.savefig(output_path, dpi=300)
    logger.success(f"Survived Distribution plot created successfully (Path: {output_path})")

    columns = ['age_group','pclass','sex','embarked','sibsp','parch','family_size','is_alone']
    for col in columns:
        count_plot(col=col,orientation='vertical',data=df.loc[df['survived']==1])
        output_path = os.path.join(DESTINATION, f"{col}")
        plt.savefig(output_path, dpi=300)
        logger.success(f"{col} plot created successfully (Path: {output_path})")
    logger.success(f"All plots are created successfully (Path: {DESTINATION})")


