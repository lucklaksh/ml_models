import uvicorn

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import pickle
import pandas as pd
import numpy as np


app = FastAPI()
pickle_in = open("../../kaggle/mnist_trained_model.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.get('/')
def index():
    return {'message': 'Hello, stranger'}


@app.get('/{name}')
def get_name(name:str):
    return {"Welcome To Lucky's Class": f"{name}"}


@app.post("/predictimages/")
async def predict_images(images: List[UploadFile]):
    predlist = []
    for image in images:
        img = Image.open(image.file)
        # Perform image processing or save the image here
        img = img.resize((28, 28))  # Resize the image to fit within 28*28 pixels
        img_gray = img.convert("L")
        #
        # image_data = list(img_gray.getdata())
        # predicted_class = classifier.predict([image_data])[0]

        img_array = np.array(img_gray)
        brightness_1d = img_array.flatten()
        brightness_1d = brightness_1d/255.0
        brightness_1d = pd.DataFrame(brightness_1d.reshape(-1, len(brightness_1d)))
        predicted_value = classifier.predict(brightness_1d)
        print(predicted_value)
        predlist.append(predicted_value)

        img_gray.save(f"uploaded_{image.filename}")
    return JSONResponse(content={"message": f"{predlist}"})
