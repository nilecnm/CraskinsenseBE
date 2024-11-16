import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy      as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import base64
import albumentations as A
from PIL import Image


os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow.keras.models import load_model
import segmentation_models as sm

# fast api
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost:3000",  # frontend ที่รันอยู่บน localhost:3000
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตเฉพาะโดเมนที่ระบุใน origins
    allow_credentials=True,
    allow_methods=["*"],  # อนุญาตทุก HTTP methods 
    allow_headers=["*"],  # อนุญาตทุก headers
)

BACKBONE = 'vgg19'
n_classes = 1
activation = 'sigmoid'
model_test = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

weights_path = './best_model.weights.h5'
model_test.load_weights(weights_path)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def prepare_image_for_prediction(image, target_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image  # Return the image without adding batch dimension

def convert_to_base64(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

target_size = (256, 256)  # Model input size

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)

    prepared_image = prepare_image_for_prediction(image, target_size)

    pred = model_test.predict(np.expand_dims(prepared_image, axis=0))
    pred_mask = (pred[0, :, :, 0] > 0.5).astype(np.uint8)

    pred_mask = pred_mask * 255

    pred_mask_base64 = convert_to_base64(pred_mask)

    return JSONResponse(content={"predicted_mask_base64": pred_mask_base64})