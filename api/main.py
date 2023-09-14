from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy", "Non-Potato Leaf"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())

        # Check image dimensions
        if image.shape[:2] != (256, 256):
            return {
                'error': 'Invalid Image Size',
                'message': 'Please enter a 256x256 image.'
            }

        img_batch = np.expand_dims(image, 0)
        
        predictions = MODEL.predict(img_batch)

        predicted_index = np.argmax(predictions[0])
        if predicted_index < len(CLASS_NAMES):
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = np.max(predictions[0])
            return {
                'class': predicted_class,
                'confidence': float(confidence)
            }
        else:
            return {
                'class': 'Not Classified',
                'confidence': 0.0
            }
    except Exception as e:
        return {
            'error': 'Internal Server Error',
            'message': str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)