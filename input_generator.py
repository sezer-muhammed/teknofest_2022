import cv2
import numpy as np
import time
import logging
import requests

from TAKIM_BAGLANTI_ARAYUZU.src.connection_handler import ConnectionHandler
from pathlib import Path

class Input_generator():
    def __init__(self, extension = "", source = "", team_name = "", password = "", evaluation_server_url = "") -> None:
        self.extension = extension
        self.source = source
        self.video_player_loop_skip = 0

        if self.extension == "":
            print("Uzanti Tipi Girin!!")
            exit()

        if self.extension == "tekno":
            self.server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)
            self.evaluation_server_url = evaluation_server_url
            self.frames_json = self.server.get_frames()
            self.images_folder = "./_images/"
            Path(self.images_folder).mkdir(parents=True, exist_ok=True)
            self.counter = 0

        if self.extension == "mp4":
            self.cam = cv2.VideoCapture(self.source)

    @staticmethod
    def teknofest_loop(img_url, images_folder):
        t1 = time.perf_counter()
        img_bytes = requests.get(img_url).content
        image_name = img_url.split("/")[-1]  # frame_x.jpg

        with open(images_folder + image_name, 'wb') as img_file:
            img_file.write(img_bytes)

        t2 = time.perf_counter()

        logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')
        return cv2.imread(images_folder + image_name)

    def mp4_loop(self):
        while True:

            self.video_player_loop_skip += 1
            if self.video_player_loop_skip % 8 == 0:
                break
            self.ret, self.frame = self.cam.read()
            self.color_wrap()
            if self.ret == False:
                exit()

    def get_frame(self):
        if self.extension == "mp4":
            self.mp4_loop()
            return self.frame, ""
        if self.extension == "tekno":
            if self.counter == len(self.frames_json):
                exit()
            frame_name = self.frames_json[self.counter]
            self.counter += 1
            self.frame = self.teknofest_loop(self.evaluation_server_url + "media" + frame_name["image_url"], "./_images/")
            self.color_wrap()
            return self.frame, frame_name

    def color_wrap(self):
        pass