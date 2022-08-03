from general_utils import nms
from input_generator import Input_generator
from TAKIM_BAGLANTI_ARAYUZU.src.constants import classes, landing_statuses
from model_zoo import yolov5_inference
from pathlib import Path
import time
from decouple import config
import cv2
from histogram import *
import yaml
from visual_utils import visualiser

parameter_data = "teknofest" #For parameters.yaml file
parameter_model = "yolov5" #For parameters.yaml file


def configure_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(filename=log_filename, level=logging.INFO, filemode="w",
                        format='%(asctime)s - %(levelname)s - %(message)s')


with open("parameters.yaml", "r") as stream:
    parameters = yaml.safe_load(stream)
    extension = parameters["input"]["extension"]
    histogram_equalize = parameters["preprocess"]["normalization"]
    clahe_ratio = parameters["preprocess"]["clahe_ratio"]
    video_source = parameters["input"]["video_source"]
    nms_thres = parameters[parameter_model]["iou_thres"]
    confs = parameters["conf_mat"]

config.search_path = "./TAKIM_BAGLANTI_ARAYUZU/config/"
team_name = config('TEAM_NAME')
password = config('PASSWORD')
evaluation_server_url = config("EVALUATION_SERVER_URL")

images_folder = "./_images/"
Path(images_folder).mkdir(parents=True, exist_ok=True)
configure_logger(team_name)


workflow = "flow_three" #!Check it.
model = yolov5_inference(workflow)
input_generator = Input_generator(extension, video_source, team_name=team_name, password=password, evaluation_server_url=evaluation_server_url)
ressam_bob = visualiser(parameter_data, parameter_model, resolution=1080)



while True:
    frame, frame_name = input_generator.get_frame() #! BGR Frame
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

    if extension == "tekno":
        send_predictions_from_result(frame_name, output, input_generator, classes=classes, landing_statuses=landing_statuses)

    frame_to_show = ressam_bob.insert_labes(frame, output)

    cv2.imshow("frame", cv2.resize(frame_to_show, (1280+320, 960)))
    cv2.waitKey(1)