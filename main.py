# -*- coding: utf-8 -*-
import cv2
import numpy              as np
import numpy              as np
import base64
from PIL                  import Image

# fast api
from io                      import BytesIO
from fastapi                 import FastAPI, File, UploadFile
from fastapi.responses       import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model import predict as predict_model

"""**API Section**"""

app     = FastAPI(root_path="/api")
origins = ["http://localhost:3000"]  # frontend ที่รันอยู่บน localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # อนุญาตเฉพาะโดเมนที่ระบุใน origins
    allow_credentials=True,
    allow_methods=["*"],             # อนุญาตทุก HTTP methods
    allow_headers=["*"],)            # อนุญาตทุก headers

def convert_to_base64(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, target_size)
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image       = Image.open(BytesIO(image_bytes)).convert("RGB")
    image       = np.array(image)

    avg_value, max_value, min_value, overlayed_image, density_plot, number_point = predict_model(image)

    return JSONResponse(content={
        "avg_value": float(avg_value),
        "max_value": float(max_value),
        "min_value": float(min_value),
        "overlayed_image": convert_to_base64(overlayed_image),
        "density_plot": convert_to_base64(density_plot),
        "number_point": float(number_point),
    })
