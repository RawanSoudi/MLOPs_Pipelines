import numpy as np
import litserve as ls
import pickle
import pandas as pd
from src.deployment.online.requests import InferenceRequest

class InferenceAPI(ls.LitAPI):
    def setup(self, device = "cpu"):
        with open("models/lr_model.pkl", "rb") as pkl:
            self._model = pickle.load(pkl)

    def decode_request(self, request):
        try:
            columns = ["PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
            InferenceRequest(**request["input"])
            df = pd.DataFrame([request["input"]])
            df.columns = [x.lower() for x in df.columns]
            df['family_size'] = df['sibsp'] + df['parch']
            df['is_alone'] = df['family_size'].apply(lambda x: 1 if x == 0 else 0)
            df.loc[(df['age'] >= 0)  & (df['age'] < 16),  'age_group'] = 'Below 16'
            df.loc[(df['age'] >= 16)  & (df['age'] < 40),  'age_group'] = 'Between 16-40'
            df.loc[(df['age'] >= 40)  & (df['age'] < 60),  'age_group'] = 'Between 40-60'
            df.loc[(df['age'] >= 60),  'age_group'] = 'Above 60'
            df.set_index("passengerid",inplace=True)
            x = df.loc[:,["pclass","sex","age","fare","embarked","family_size","is_alone"]]
            x['sex'] = x['sex'].map({'male':0,'female':1})
            x['embarked'] = x['embarked'].map({'S':1,'C':2,'Q':3})
            x = np.asarray(x)
            return x
        except:
            return None

    def predict(self, x):
        if x is not None:
            return self._model.predict(x)
        else:
            return None

    def encode_response(self, output):
        if output is None:
            message = "Error Occurred"
        else:
            message = "Response Produced Successfully"
        return {
            "message": message,
            "prediction": output.tolist()
        }
    
