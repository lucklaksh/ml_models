from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import numpy as np

app = FastAPI()


@app.post("/uploadimages/")
async def upload_images(images: List[UploadFile]):
    for image in images:
        img = Image.open(image.file)
        # Perform image processing or save the image here
        img = img.resize((28, 28))  # Resize the image to fit within 28*28 pixels
        img_gray = img.convert("L")

        img_array = np.array(img_gray)

        brightness_1d = img_array.flatten()

        print(len(brightness_1d))
        img_gray.save(f"uploaded_{image.filename}")
    return JSONResponse(content={"message": "Images uploaded and processed"})

