# -*- coding: utf-8 -*-
import cv2
import numpy              as np
import matplotlib.pyplot  as plt
import os
import numpy              as np
import tensorflow         as tf
from tensorflow           import keras
import matplotlib.pyplot  as plt
import base64
import albumentations     as A
from PIL                  import Image
import pandas as pd


os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow.keras.models import load_model
import segmentation_models   as sm

# fast api
from io                      import BytesIO
from fastapi                 import FastAPI, File, UploadFile
from fastapi.responses       import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

"""**API Section**"""

app     = FastAPI()
origins = ["http://localhost:3000"]  # frontend ที่รันอยู่บน localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # อนุญาตเฉพาะโดเมนที่ระบุใน origins
    allow_credentials=True,
    allow_methods=["*"],             # อนุญาตทุก HTTP methods
    allow_headers=["*"],)            # อนุญาตทุก headers

"""**Image Preprocessing**"""

def prepare_image_for_prediction(image, target_size):
    image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # ทำการเรียกข้อมูลรูปภาพ
    ori_size = image.shape[:2]                             # เก็บขนาด Original ของรูปภาพเอาไว้
    image    = cv2.resize(image, target_size)              # ปรับขนาดรูปภาพให้เป็นขนาดที่เท่ากับรูปภาพที่ควรเข้า Model
    return image,ori_size                                  # ส่ง Image และค่าขนาด Original Size

"""**Setup & Loading Model**"""

BACKBONE     = 'vgg19'
n_classes    = 1
activation   = 'sigmoid'
model_test   = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
weights_path = './best_model.weights.h5'
model_test.load_weights(weights_path)

target_size = (256, 256)

"""**Prediction Function**"""

def prediction_model(image,original_size,model_test):
    kernel       = np.ones((18,18), np.uint8)                                  # ทำการกำหนด Kernal สำหรับลด Noise ของการ Prediction Model
    pred         = model_test.predict(np.expand_dims(image, axis=0))           # ทำการ Prediction
    pred_mask    = (pred[0, :, :, 0] > 0.5).astype(np.uint8)                   # เปลี่ยน Mask ให้เป็น Binary Mask พื้นหลังดำส่วน Label ขาว
    re_mask      = cv2.resize(pred_mask, (original_size[1], original_size[0])) # ทำการ Resize Predicted Mask ให้มีขนาดเท่ากับภาพ Original
    dilated_mask = cv2.morphologyEx(re_mask, cv2.MORPH_OPEN, kernel)           # ทำการปรับแต่งรูปภาพเพิ่มเติมภายหลังการ Prediction
    return dilated_mask

def calculate_thickness(image, mask):
    # Convert the mask to grayscale and then apply binary threshold
    if (mask.shape == 3):
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Compute the distance transform
    distance_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # Extract non-zero distances and sort them
    non_zero_distances = distance_transform[distance_transform > 0]
    sorted_distances = np.sort(non_zero_distances)[::-1]

    # Select top 5% values
    top_5_percent_values = sorted_distances[:int(0.005 * len(sorted_distances))]

    # Calculate average, max, and min values
    avg_value = np.mean(top_5_percent_values)
    max_value = np.max(top_5_percent_values)
    min_value = np.min(top_5_percent_values)
    avg_value = round(avg_value, 2)
    max_value = round(max_value, 2)
    min_value = round(min_value, 2)

    # Normalize the distance transform for visualization
    norm_distance_transform = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_distance_transform, cv2.COLORMAP_JET)

    # Mask out the black areas in the heatmap (where the distance transform is zero)
    heatmap[distance_transform == 0] = (0, 0, 0)  # Set black regions to black
    original_colored = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlayed_image = cv2.addWeighted(original_colored, 1, heatmap, 0.6, 0)

    #ขั้นตอนในการ Plot
    distibution_value = pd.DataFrame(top_5_percent_values, columns=["Thickness"])
    distibution_arr = distibution_value["Thickness"].to_list()

    return avg_value, max_value, min_value, distibution_value, overlayed_image, distibution_arr

def convert_to_base64(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image       = Image.open(BytesIO(image_bytes)).convert("RGB")
    image       = np.array(image)

    prepared_image,original_size = prepare_image_for_prediction(image, target_size)
    pred_mask                    = prediction_model(prepared_image,original_size,model_test)
    avg_value, max_value, min_value, distibution_value,overlayed_image, distibution_value = calculate_thickness(image, pred_mask)
    pred_mask_base64             = convert_to_base64(overlayed_image)

    return JSONResponse(content={
        "predicted_mask_base64": pred_mask_base64,
        "distibution_value": distibution_value,
        "avg_value": avg_value,
        "max_value": max_value,
        "min_value": min_value,
    })
