# -*- coding: utf-8 -*-
"""Another copy of Edited Code.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PqTQFohaRciNBYaf1Py-QpquJXiDP-W8

1.อ่านภาพเข้าโปรแกรม
ใช้ cv2.imread เพื่ออ่านภาพจากไฟล์ชื่อ "1H_UV_cont_1.tif" แล้วเก็บในตัวแปร image_input
กำหนดเส้นทางไฟล์โมเดลที่ผ่านการเทรน

2.กำหนดเส้นทางไฟล์น้ำหนักของโมเดล (weights) ในตัวแปร weight_path เช่น "/content/best_model_resnet50.weights.h5"
ตรวจจับ ArUco Markers และตัดส่วนของภาพ

3.ใช้ฟังก์ชัน detection_aruco เพื่อค้นหาและระบุ ArUco markers ในภาพ จากนั้นตัดภาพตามตำแหน่งที่ตรวจจับได้ และเก็บไว้ในตัวแปร crop_image
เตรียมภาพที่ถูกตัดเพื่อใช้ในการพยากรณ์

4.ใช้ฟังก์ชัน prepare_image_for_prediction ทำการปรับขนาดและปรับค่าพิกเซลของภาพให้เหมาะสมกับโมเดล โดยรับภาพ crop_image และปรับขนาดเป็น (256, 256)
คืนค่าภาพที่ปรับแล้วในตัวแปร image และภาพต้นฉบับที่ใช้ร่วมในกระบวนการพยากรณ์เก็บในตัวแปร original
ทำการพยากรณ์

5.ใช้ฟังก์ชัน prediction_model โดยป้อน image, original และน้ำหนักโมเดลจาก weight_path เพื่อสร้างผลลัพธ์ pre_mask
คำนวณค่าความหนาและผลลัพธ์ที่เกี่ยวข้อง

6.ใช้ฟังก์ชัน calculate_thickness โดยป้อนภาพที่ถูกตัด (crop_image) และหน้ากากที่พยากรณ์ได้ (pre_mask) เพื่อคำนวณ:
ค่าเฉลี่ย (avg_value)
ค่าสูงสุด (max_value)
ค่าต่ำสุด (min_value)
ภาพที่ซ้อนผลลัพธ์ (overlayed_image)
กราฟความหนา (density_plot)
จำนวนจุดที่ตรวจจับได้ (number_point)
แสดงผลลัพธ์ที่คำนวณได้


**# ไม่จำเป็นในหน้าแอพ**
แสดงค่าเฉลี่ย (avg_value), ค่าสูงสุด (max_value), ค่าต่ำสุด (min_value) และจำนวนจุด (number_point) ผ่าน print
แสดงภาพที่เกี่ยวข้อง

ใช้ Matplotlib แสดง:
ภาพที่ซ้อนผลลัพธ์ (overlayed_image) ในกรอบซ้าย
กราฟความหนา (density_plot) ในกรอบขวา โดยใช้ colormap 'viridis' หรือจัดรูปแบบถ้าเป็น matplotlib figure

**Install package**
"""

import os
import cv2
import numpy      as np
import tensorflow as tf
from   tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas            as pd
import albumentations    as A
from   PIL             import Image
from   io              import BytesIO
from   matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# Set environment variable
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import seaborn as sns
import base64

"""# **Define Function**

**Image Capture Processing Function**
ฟังก์ชันสำหรับการ Crop รูปภาพที่ถ่ายจากมือถือเพื่อให้ได้แค่พื้นที่ที่เราสนใจเท่านั้น
"""

# Path to pre-trained model weights
weight_path = "./best_model_resnet50.weights.h5"

def get_marker_centers(corners):
    """
    Get the center points of all detected ArUco markers.
    """
    centers = []
    for marker_corner in corners:
        marker_corner = marker_corner.reshape((4, 2))
        topLeft, topRight, bottomRight, bottomLeft = marker_corner
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        centers.append((cX, cY))
    return np.array(centers, dtype=np.float32)

def warp_perspective_to_quad(image, quad_points):
    (tl, tr, br, bl) = quad_points
    width_top        = np.linalg.norm(tr - tl)
    width_bottom     = np.linalg.norm(br - bl)
    max_width        = int(max(width_top, width_bottom))

    height_left      = np.linalg.norm(bl - tl)
    height_right     = np.linalg.norm(br - tr)
    max_height       = int(max(height_left, height_right))
    dst_points       = np.array([[0, 0],[max_width - 1, 0],[max_width - 1, max_height - 1],[0, max_height - 1]], dtype=np.float32)
    matrix           = cv2.getPerspectiveTransform(quad_points, dst_points)
    warped           = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def detection_aruco(image):
    aruco_type = "DICT_4X4_100"
    ARUCO_DICT = {"DICT_4X4_100": cv2.aruco.DICT_4X4_100}
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    parameters = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(dictionary, parameters)
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image)
    if markerCorners and len(markerCorners) >= 4:
      marker_centers = get_marker_centers(markerCorners)
      sum_coords     = marker_centers.sum(axis=1)
      diff_coords    = np.diff(marker_centers, axis=1)
      topLeft        = marker_centers[np.argmin(sum_coords)]
      bottomRight    = marker_centers[np.argmax(sum_coords)]
      topRight       = marker_centers[np.argmin(diff_coords)]
      bottomLeft     = marker_centers[np.argmax(diff_coords)]
      quad_points    = np.array([topLeft, topRight, bottomRight, bottomLeft], dtype=np.float32)
      cropped_quad   = warp_perspective_to_quad(image, quad_points)
      cropped_rgb    = cv2.cvtColor(cropped_quad, cv2.COLOR_BGR2RGB)
      return cropped_rgb
    else:
      return image

"""**Preprocessing Image**"""

def prepare_image_for_prediction(image, target_size):
    image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # ทำการเรียกข้อมูลรูปภาพ
    ori_size = image.shape[:2]                             # เก็บขนาด Original ของรูปภาพเอาไว้
    image    = cv2.resize(image, target_size)              # ปรับขนาดรูปภาพให้เป็นขนาดที่เท่ากับรูปภาพที่ควรเข้า Model
    return image,ori_size                                  # ส่ง Image และค่าขนาด Original Size

def load_model(weights_path):
    BACKBONE     = 'resnet50'
    n_classes    = 1
    activation   = 'sigmoid'
    model_test   = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    model_test.load_weights(weights_path)
    return model_test

def prediction_model(image,original_size,weights_path):
    #kernel       = np.ones((15,15), np.uint8)
    model_test   = load_model(weights_path)                                    # ทำการกำหนด Kernal สำหรับลด Noise ของการ Prediction Model
    pred         = model_test.predict(np.expand_dims(image, axis=0))           # ทำการ Prediction
    pred_mask    = (pred[0, :, :, 0] > 0.5).astype(np.uint8)                   # เปลี่ยน Mask ให้เป็น Binary Mask พื้นหลังดำส่วน Label ขาว
    re_mask      = cv2.resize(pred_mask, (original_size[1], original_size[0])) # ทำการ Resize Predicted Mask ให้มีขนาดเท่ากับภาพ Original
    #dilated_mask = cv2.morphologyEx(re_mask, cv2.MORPH_OPEN, kernel)           # ทำการปรับแต่งรูปภาพเพิ่มเติมภายหลังการ Prediction
    return re_mask

def plot_density(top_5_percent_values):
    data = pd.DataFrame({
        'thickness_x': top_5_percent_values,
        'thickness_y': np.array(top_5_percent_values) + np.random.normal(0, 5, len(top_5_percent_values))})

    g = sns.jointplot(data=data, x="thickness_x", y="thickness_y", kind="reg", color='green')
    g.ax_joint.collections[0].set_facecolor('blue')
    g.ax_joint.collections[1].set_edgecolor('green')
    fig = g.fig
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    image_np = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    #print(f"Image shape: {image_np.shape}")
    return image_np

def calculate_thickness(image, mask):
    _, binary_mask       = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    distance_transform   = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    non_zero_distances   = distance_transform[distance_transform > 0]
    sorted_distances     = np.sort(non_zero_distances)[::-1]
    top_5_percent_values = sorted_distances[:int(0.005 * len(sorted_distances))]
    number_point = len(top_5_percent_values)
    avg_value    = round(np.mean(top_5_percent_values),2)
    max_value    = round(np.max(top_5_percent_values),2)
    min_value    = round(np.min(top_5_percent_values),2)
    norm_distance_transform = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap                 = cv2.applyColorMap(norm_distance_transform, cv2.COLORMAP_JET)
    heatmap[distance_transform == 0] = (0, 0, 0)
    original_colored = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlayed_image  = cv2.addWeighted(original_colored, 1, heatmap, 0.6, 0)
    overlayed_image  = cv2.resize(overlayed_image, (256,256))
    density_plot     = plot_density(top_5_percent_values)
    return avg_value, max_value, min_value,overlayed_image,density_plot,number_point

def predict(image_input):
    crop_image = detection_aruco(image_input)

    # Prepare the cropped image for prediction (resize and normalize)
    image, original = prepare_image_for_prediction(crop_image, target_size=(256, 256))

    # Perform prediction using the pre-trained model
    pre_mask = prediction_model(image, original, weight_path)

    # Calculate thickness and generate outputs
    # avg_value, max_value, min_value, overlayed_image, density_plot, number_point = calculate_thickness(crop_image, pre_mask)
    return calculate_thickness(crop_image, pre_mask)