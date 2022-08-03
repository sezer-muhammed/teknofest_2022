from general_utils import nms
from input_generator import Input_generator
from model_zoo import yolov5_inference
from pathlib import Path
import time
import cv2
from histogram import *
import yaml
from visual_utils import visualiser
from iou_w_jit import *
from general_utils import upload_image, upload_xml_labels
import glob
import pandas as pd
import random

parameter_data = "teknofest" #For parameters.yaml file
parameter_model = "yolov5" #For parameters.yaml file

folder = glob.glob("data/*.xml")
random.shuffle(folder)

with open("parameters.yaml", "r") as stream:
    parameters = yaml.safe_load(stream)
    extension = parameters["input"]["extension"]
    histogram_equalize = parameters["preprocess"]["normalization"]
    clahe_ratio = parameters["preprocess"]["clahe_ratio"]
    video_source = parameters["input"]["video_source"]
    nms_thres = parameters[parameter_model]["iou_thres"]
    flows = parameters["performance_flows"]
    classes = parameters[parameter_data]["names_cls"]
    flow_list_all = parameters["work_flow"]["flow_calibrator"]

#* Model Type: raw, sahi | conf_thres | Sahi Slice Ratio | Classes | Raw Input Bool

topics = ["Raw or SAHI", "Confidence", "Sahi Slice Ratio", "Class", "Success Rate", "Sahi Raw Input"]

flow = "flow_calibrator"


for flow_list in flow_list_all:
    model = yolov5_inference(flow, flow_array=[flow_list])
    clss_performances = [iou_optimizer() for i in flow_list[3]]

    for i, xml_file in enumerate(folder):
        try:
            Ground_truth = upload_xml_labels(xml_file, parameter_data) #Returns detections as np array shape = (N, 5), [xmin, ymin, xmax, ymax, classid]
            frame = upload_image(xml_file, search_npy=False)
            print(f"Data Loaded {i}", end = "\r")
        except:
            continue
        output = model.detect(frame)
        for cls in flow_list[3]:
            loop_output = output[output[:, 5] == cls]
            loop_Ground_truth = Ground_truth[Ground_truth[:, 4] == cls]
            clss_performances[cls].add_sample(loop_Ground_truth, loop_output)
    infograf = []

    for i, opt in enumerate(clss_performances):
        histogram = opt.create_histogram(flow, inform=False)
        confidences = np.array(list(range(100))) / 100
        clss_id = i
        model_type = flow_list[0]
        for loop in range(100):
            infograf.append([model_type, confidences[loop], flow_list[2], classes[clss_id], histogram[loop], flow_list[4]])
    print(f"created data frame {flow_list}")

    infograf = np.array(infograf)

    try:
        past_data = pd.read_csv("Data_Raw.csv")
        dataframe = pd.DataFrame(infograf, columns=topics)
        final_data = pd.concat((past_data, dataframe))
        final_data.to_csv("Data_Raw.csv", index = False)
        print("Appended")
    except:
        dataframe = pd.DataFrame(infograf, columns=topics)
        dataframe.to_csv("Data_Raw.csv", index = False)
        print("Created")
