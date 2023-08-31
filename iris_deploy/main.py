import uvicorn
from fastapi import FastAPI
from models import Iris
import numpy as np
import pickle
import pandas as pd


app = FastAPI()
pickle_in = open("../../kaggle/mnist_trained_model.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.get('/')
def index():
    return {'message': 'Hello, stranger'}


@app.get('/{name}')
def get_name(name:str):
    return {"Welcome To Lucky's Class": f"{name}"}


@app.post('/predict')
def predict_species(data:Iris):
    data = data.dict()
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']
    prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if prediction == 0:
        val = 'setosa'
    elif prediction == 1:
        val = 'versicolor'
    else:
        val = 'virginica'
    return {
        'prediction': val
    }
