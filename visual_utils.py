import cv2
import random
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import yaml
from general_utils import upload_image

class visualiser():
    def __init__(self, dataset, Model, resolution = 1080, dpi = 300):
        with open("parameters.yaml", "r") as stream:
            self.parameters = yaml.safe_load(stream)[dataset]
        with open("parameters.yaml", "r") as stream:
            other_parameters = yaml.safe_load(stream)
        self.weights = other_parameters[Model]["weights"]


        now = datetime.now()
        self.dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
        self.colors = self.parameters["colors_cls"]
        self.fig = plt.figure(figsize=(13, 9), dpi=dpi)
        self.resolution = resolution
        self.dpi = dpi
        self.images_list = {}
        self.infos = {}
    
    def reset(self):
        self.image_list = {}
        self.infos = {}
        #self.fig.clf()
        plt.clf()
        now = datetime.now()
        self.dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")

    def save_img(self, image, name):
        h, w, c = image.shape
        new_h = self.resolution
        new_w = int(new_h*w/h)
        self.images_list[name] = cv2.resize(image, (new_w, new_h))

    def list_images(self):
        return self.images_list.keys()

    def draw_images(self, names = None, show = False):
        
        if names == None:
            names = self.images_list
            number_of_items = len(names)
        else:
            number_of_items = len(names)

        box_size_h = int(np.sqrt(number_of_items - 1)) + 1
        box_size_w = int((number_of_items // box_size_h))
        if box_size_w < (number_of_items / box_size_h):
            box_size_w = box_size_w + 1
        
        plt.figure(figsize=(13, 9), dpi=self.dpi)

        for i, name in enumerate(names):
            plt.subplot(box_size_w, box_size_h, i+1)
            plt.imshow(cv2.cvtColor(self.images_list[name], cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(name)
        plt.subplots_adjust(wspace=0.02, hspace=0.1)
        plt.savefig("figures/" + self.dt_string + ".jpg", bbox_inches='tight')
        
        if show:
            figure = upload_image("figures/" + self.dt_string + ".jpg")
            return figure
        return None

    def insert_labes(self, image, labels, name = None, save_img = False):
        image = image.copy()
        
        try:
            _, model_sign = labels.shape
        except:
            model_sign = 5
        model_sign = model_sign - 5
        
        for label in labels:
            image = cv2.rectangle(image, (int(label[0]), int(label[1])), (int(label[2]), int(label[3])), tuple(self.colors[int(label[4 + model_sign])]), 1)
            if model_sign:
                image = cv2.putText(image, self.parameters["names_cls"][int(label[4 + model_sign])], (int(label[0]), int(label[1] + 35)), cv2.FONT_HERSHEY_COMPLEX, 1.4, tuple(self.colors[int(label[4 + model_sign])]), 2)
            else:
                image = cv2.putText(image, self.parameters["names_cls"][int(label[4 + model_sign])], (int(label[0]), int(label[3] - 10)), cv2.FONT_HERSHEY_COMPLEX, 1.4, tuple(self.colors[int(label[4 + model_sign])]), 2)
        
        if save_img:
            self.save_img(image, name)

        return image

    def insert_info(self, infos, name = None, save_img = False, add_model_info = True):
        self.infos.update(infos)
        self.infos["Date & Time"] = self.dt_string
        if add_model_info:
            self.infos["Model"] = self.weights
        image = np.zeros((1080, 1920, 3), np.uint8)
        keys = self.infos.keys()
        cv2.line(image, (0, 10), (1920, 10), (255, 255, 255), 2)
        for i, key in enumerate(keys):
            i = i + 1
            value = self.infos[key]
            cv2.putText(image, key, (5, i * 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)
            if isinstance(value, str):
                lenght = cv2.getTextSize(key, cv2.FONT_HERSHEY_COMPLEX, 1.5, 2)[0][0]
                cv2.putText(image, str(value), (100 + lenght, i * 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
            else:
                cv2.putText(image, str(round(value, 5)), (1000, i * 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
            cv2.line(image, (0, i * 50 + 10), (1920, i * 50 + 10), (255, 255, 255), 2)

        if save_img:
            self.save_img(image, name)

        return image

    def insert_difference(self, image, negatives, model_output, name = None, save_img = False):
        try:
            outputs = model_output[np.where(negatives != 0)[0]]
        except:
            outputs = []
        image = self.insert_labes(image, outputs)
        if save_img:
            self.save_img(image, name)
