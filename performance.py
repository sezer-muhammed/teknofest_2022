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

parameter_data = "teknofest" #For parameters.yaml file
parameter_model = "yolov5" #For parameters.yaml file


folder = glob.glob("data/*.xml")


with open("parameters.yaml", "r") as stream:
    parameters = yaml.safe_load(stream)
    extension = parameters["input"]["extension"]
    histogram_equalize = parameters["preprocess"]["normalization"]
    clahe_ratio = parameters["preprocess"]["clahe_ratio"]
    video_source = parameters["input"]["video_source"]
    nms_thres = parameters[parameter_model]["iou_thres"]
    flows = parameters["performance_flows"]
    confs = parameters["conf_mat"]

ressam_bob = visualiser(parameter_data, parameter_model, resolution=1080)

for flow in flows:
    model = yolov5_inference(flow)
    performancer = iou_optimizer()
    indexes = list(range(101))
    print("Starting to calculate perforce for", flow)
    for i, xml_file in enumerate(folder):
        Ground_truth = upload_xml_labels(xml_file, parameter_data) #Returns detections as np array shape = (N, 5), [xmin, ymin, xmax, ymax, classid]
        frame = upload_image(xml_file, search_npy=False)

        if histogram_equalize:
            frame = equalize(frame, clahe_ratio)

        output = model.detect(frame)

        output = regularize_landing_statues(output, frame)

        output = nms(output, nms_thres + 0.02)

        conf_filtered_output = np.empty((0, 6))
        for ii, conf in enumerate(confs):
            filter = output[output[:, 5] == ii]
            filter = filter[filter[:, 4] > conf]
            if filter.size != 0:
                conf_filtered_output = np.concatenate((conf_filtered_output, filter))
        output = conf_filtered_output

        frame_to_show = ressam_bob.insert_labes(frame, output)

        performancer.add_sample(Ground_truth, output)

        print(i, end = "\r")
    print("Done!")
    performancer.create_histogram(flow)