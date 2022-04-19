from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pydantic import BaseModel
from numpy import asarray
import urllib.request

app = FastAPI()

MODEL = tf.keras.models.load_model('./saved_models/1')

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

class img_url(BaseModel):
    url: str

    class Config:
        schema_extra = {
            "example": {
                "url" : "https://i.ibb.co/FBSztPS/0120.jpg"
            }
        }


@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_without_alpha = image[:,:,:3]
    img_batch = np.expand_dims(image_without_alpha, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

@app.post(
    "/"
)
async def classify_url(item: img_url):
    req = urllib.request.urlretrieve(item.url, "saved")
    image = Image.open("saved")
    data = asarray(image)
    image_without_alpha = data[:,:,:3]
    img_batch = np.expand_dims(image_without_alpha, 0)

    predictions = MODEL.predict(img_batch)    
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)