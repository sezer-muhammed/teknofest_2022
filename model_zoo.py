from pathlib import Path
from time import time

import cv2
import numpy as np
import torch
from sahi.model import Yolov5DetectionModel
from sahi.predict import get_sliced_prediction
import time
import yaml


class yolov5_inference():
    def __init__(self, workflow, flow_array = None):
        with open("parameters.yaml", "r") as stream:
            parameters = yaml.safe_load(stream)
            if flow_array == None:
                self.work_flow = parameters["work_flow"][workflow]
                print(self.work_flow)
            else:
                self.work_flow = flow_array
                print(self.work_flow)
            time.sleep(2)
            parameters = parameters["yolov5"]
        

        self.weights = parameters["weights"]
        self.iou_thres = parameters["iou_thres"]
        self.img_size = parameters["img_size"]
        self.agnostic_nms = True

        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weights)
        self.model.conf = 0.01
        self.model.agnostic = self.agnostic_nms

        self.detection_model = Yolov5DetectionModel(
                                model_path=self.weights,
                                confidence_threshold=0.01,
                                device="cuda:0"
                            )
            


    def detect(self, img):
        """
        Args:
            img ([np array]): [BGR Image]

        Returns:
            [type]: [Nx6 array | xmin, ymin, xmax, ymax, conf, class]
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        final_output = np.empty((0, 6))
        #Model Type: raw, sahi | conf_thres | Sahi Slice Ratio | Classes | Raw Input Bool
        for step in self.work_flow:
            if step[0] == "raw":
                output = self.detect_raw(img, step)
            elif step[0] == "sahi":
                output = self.detect_sahi(img, step)
            final_output = np.concatenate((final_output, output), axis = 0)
        return final_output


    def detect_raw(self, img, step):
        result = self.model(img, size = self.img_size)
        det = result.xyxy[0].cpu().numpy()
        if det.size == 0:
            return np.empty((0, 6))

        mask = np.isin(det[:, 5], step[3])

        det = det[mask, :]

        if det.size == 0:
            return np.empty((0, 6))       
        return det

    def detect_sahi(self, img, step):
        h, w, c = img.shape
        result = get_sliced_prediction(
                    img,
                    self.detection_model,
                    slice_height = int(step[2] * h),
                    slice_width = int(step[2] * h),
                    overlap_height_ratio = 0.35,
                    overlap_width_ratio = 0.35,
                    perform_standard_pred=step[4],
                    postprocess_class_agnostic=self.agnostic_nms,
                    verbose=0
                )
        
        det = []
        
        for object in result.object_prediction_list:
            det.append([object.bbox.minx, object.bbox.miny, object.bbox.maxx, object.bbox.maxy, object.score.value, object.category.id])
        
        det = np.array(det)
        if det.size == 0:
            return np.empty((0, 6))

        mask = np.isin(det[:, 5], step[3])

        det = det[mask, :]

        if det.size == 0:
            return np.empty((0, 6))

        return det
