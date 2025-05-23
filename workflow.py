import pandas as pd
import duckdb
import mlflow.sklearn
import pickle
from prefect import flow, task
import os
import dagshub
from dotenv import load_dotenv

load_dotenv()
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DUCKDB_CONN = f"md:titanic_test?motherduck_token={MOTHERDUCK_TOKEN}"
MODEL_NAME = "DT_Titanic"
MODEL_ALIAS = "1"
PREDICTION_TABLE = "predictions"

dagshub.auth.add_app_token(os.getenv("DAGSHUB_TOKEN"))




@task
def extract() -> pd.DataFrame:
    con = duckdb.connect(DUCKDB_CONN)
    query = "SELECT * FROM test;"
    df = con.execute(query).fetchdf()
    return df


@task
def transform(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [x.lower() for x in df.columns]
    df['family_size'] = df['sibsp'] + df['parch']
    df['is_alone'] = df['family_size'].apply(lambda x: 1 if x == 0 else 0)
    df.set_index("passengerid",inplace=True)
    x = df.loc[:,["pclass","sex","age","fare","embarked","family_size","is_alone"]]
    x['sex'] = x['sex'].map({'male':0,'female':1})
    x['embarked'] = x['embarked'].map({'S':1,'C':2,'Q':3})
    return df

@task
def predict(df: pd.DataFrame) -> pd.DataFrame:
    logged_model = 'runs:/79ca65c5e68545e3bafb1fe155e1b0a0/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    features = df.drop(columns=['id'], errors='ignore')
    predictions = loaded_model.predict(features)
    df['prediction'] = predictions
    return df


@task
def load(df: pd.DataFrame):
    con = duckdb.connect(DUCKDB_CONN)
    con.execute(f"CREATE TABLE IF NOT EXISTS {PREDICTION_TABLE} AS SELECT * FROM df LIMIT 0")
    con.register('df', df)
    con.execute(f"INSERT INTO {PREDICTION_TABLE} SELECT * FROM df")
    print(f"Predictions written to {PREDICTION_TABLE}")


@flow(name="Titanic Batch Job")
def titanic_batch_job():
    df = extract()
    df = transform(df)
    preds = predict(df)
    load(preds)


if __name__ == "__main__":
    titanic_batch_job()