import xml.etree.ElementTree as ET
import numpy as np
import cv2
from keras_preprocessing.image import load_img, img_to_array
import yaml


def upload_xml_labels(xml_file, dataset):

    with open("parameters.yaml", "r") as stream:
        parameters = yaml.safe_load(stream)
        cls_names = parameters[dataset]["names_cls"]

    if ".xml" in xml_file:
        pass
    else:
        xml_file = xml_file.split(".jpg")[0].split(".png")[0].split(".npy")[0] + ".xml"
    
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        ymin, xmin, ymax, xmax = None, None, None, None
        try:
            label = cls_names.index(boxes.find("name").text)
        except:
            label = 0
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax, label]
        list_with_all_boxes.append(list_with_single_boxes)


    gt = np.array(list_with_all_boxes)
    if gt.size == 0:
        return np.empty((0, 6))
    return gt

def upload_image(path, search_npy = False):
    """
    Ortalama 70 ms sürede görsel yüklüyor

    Görselin path'i input, return ettiği görsel np arrayi olarak

    uzantı .jpg, .xml, .npy veya .png olmak zorundadır
    """

    if search_npy:
        path_npy = path.split(".jpg")[0].split(".png")[0].split(".xml")[0].split(".npy")[0] + ".npy"
        try:
            return np.load(path_npy, allow_pickle=True)
        except:
            pass

    if ".npy" in path:
        return np.load(path, allow_pickle=True)
    elif ".jpg" in path or ".png" in path:
        pass
    else:
        path = path.split(".xml")[0] + ".jpg"


    image = load_img(path) 
    image = img_to_array(image, dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
        
    return dets[keep, :]