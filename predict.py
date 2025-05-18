import mlflow
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO')}"
MODEL_NAME = "DT_Titanic"
MODEL_ALIAS = "1"


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"models:/{MODEL_NAME}/{MODEL_ALIAS}")

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_ALIAS}")


test= np.array([[3, 1.0, 50.0, 0, 0, 531.0, 8.05, 146.0, 3.0]]) 

y_pred = model.predict(test)
label_map = {0: "Not Survived", 1: "Survived"}
test_pred = label_map.get(int(y_pred[0]), "Unknown")

print(f"Prediction: {y_pred[0]} ({test_pred})")