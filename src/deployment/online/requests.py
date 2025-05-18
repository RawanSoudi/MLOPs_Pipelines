from pydantic import BaseModel

class InferenceRequest(BaseModel):
    PassengerId: int
    Pclass: int
    Name: object
    Sex: object
    Age:float
    SibSp: int
    Parch: int
    Ticket: object
    Fare: float
    Cabin: object
    Embarked: object