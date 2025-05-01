from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from sklearn.model_selection import train_test_split
import os

SOURCE = os.path.join("data", "interim")
DESTINATION = os.path.join("data", "processed")

def process_data(file_name,index_col):
    input_path = os.path.join(SOURCE, f"{file_name}")
    Xtrain_path = os.path.join(DESTINATION, "processed_Xtrain.csv")
    Xtest_path = os.path.join(DESTINATION, "processed_Xtest.csv")
    ytrain_path = os.path.join(DESTINATION, "processed_ytrain.csv")
    ytest_path = os.path.join(DESTINATION, "processed_ytest.csv")
    logger.info(f"loading data from {input_path}...")

    try:
        df = pd.read_csv(input_path)
        df.columns = [x.lower() for x in df.columns]
        df.set_index(index_col,inplace=True)
        logger.success(f"Data loaded successfully (shape: {df.shape}), Index_column : {index_col}")
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
        raise typer.Exit(code=1)

    logger.info(f"Data Processing...")
    try:
        x = df.loc[:,["pclass","sex","age","fare","embarked","family_size","is_alone"]]
        y = df.loc[:,'survived']
        x['sex'] = x['sex'].map({'male':0,'female':1})
        x['embarked'] = x['embarked'].map({'S':1,'C':2,'Q':3})
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        train_mean = X_train['age'].mean()
        X_train.loc[X_train['age'].isna(),'age'] = train_mean
        X_test.loc[X_test['age'].isna(),'age'] = train_mean
        train_mode = X_train['embarked'].mode()
        X_train.loc[X_train['embarked'].isna(),'embarked'] = train_mode[0]
        X_test.loc[X_test['embarked'].isna(),'embarked'] = train_mode[0]
        X_train.to_csv(Xtrain_path,index=False)
        y_train.to_csv(ytrain_path,index=False)
        X_test.to_csv(Xtest_path,index=False)
        y_test.to_csv(ytest_path,index=False)

        logger.success(f"Data Processing done successfully (Train shape: {X_train.shape}) , (Test shape: {X_test.shape})")
        logger.success(f"XTrain saved to {Xtrain_path}")
        logger.success(f"yTrain saved to {ytrain_path}")
        logger.success(f"XTrain saved to {Xtest_path}")
        logger.success(f"yTest saved to {ytest_path}")
    except Exception as e:
        logger.error(f"Error during Data Processing: {str(e)}")
        raise typer.Exit(code=1)




