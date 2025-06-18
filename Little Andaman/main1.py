from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app"}

# Hello route
@app.get("/hello")
def say_hello(name: str = "World"):
    return {"message": f"Hello, {name}!"}

# Input model for prediction
class InputData(BaseModel):
    value: float

# Predict route (dummy logic)
@app.post("/predict")
def predict(data: InputData):
    prediction = data.value * 2  # dummy model logic
    return {"input": data.value, "prediction": prediction}