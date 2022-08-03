from TAKIM_BAGLANTI_ARAYUZU.src.frame_predictions import FramePredictions
from TAKIM_BAGLANTI_ARAYUZU.src.detected_object import DetectedObject
import cv2
import numpy as np
import time
from pathlib import Path
import logging
import concurrent.futures
import logging
from datetime import datetime

def equalize(image, ratio):

    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    clahe = cv2.createCLAHE(clipLimit = ratio, tileGridSize=(6, 3))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    final = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return final

def send_predictions_from_result(frame_name, result, input_generator, classes, landing_statuses):
    while True:
        predictions = FramePredictions(frame_name['url'], frame_name['image_url'], frame_name['video_name'])
        single_frame_result_xyxy = result

        for detection in single_frame_result_xyxy:
            if detection[5] in [0, 1, 2, 3, 4, 5]:
                landing_status = landing_statuses["Inis Alani Degil"]
                cls = classes["Tasit"]

            elif detection[5] in [6]:
                landing_status = landing_statuses["Inis Alani Degil"]
                cls = classes["Insan"]

            elif detection[5] in [7]:
                landing_status = landing_statuses["Inilebilir"]
                cls = classes["UAI"]

            elif detection[5] in [8]:
                landing_status = landing_statuses["Inilemez"]
                cls = classes["UAI"]

            elif detection[5] in [9]:
                landing_status = landing_statuses["Inilebilir"]
                cls = classes["UAP"]

            elif detection[5] in [10]:
                landing_status = landing_statuses["Inilemez"]
                cls = classes["UAP"]

            top_left_x = detection[0]
            top_left_y = detection[1]
            bottom_right_x = detection[2]
            bottom_right_y = detection[3]

            d_obj = DetectedObject(cls,
                                    landing_status,
                                    top_left_x,
                                    top_left_y,
                                    bottom_right_x,
                                    bottom_right_y)

            predictions.add_detected_object(d_obj)

        response = input_generator.server.send_prediction(predictions)
        if response.status_code != 403:
            break
        print(f"Trying to send again {frame_name['image_url']}")
        time.sleep(1)

def regularize_landing_statues(output, frame):
    h, w, c, = frame.shape
    for i, det in enumerate(output):
        if det[5] in [7, 9]:
            if det[0] < 10 or det[1] < 10 or det[2] > (w-10) or det[3] > (h-10):
                output[i][5] = det[5] + 1
    return output