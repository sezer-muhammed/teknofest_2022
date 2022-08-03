from cProfile import label
from matplotlib.pyplot import axis
import yaml
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from datetime import datetime


class iou_calculator():
    def __init__(self):
        with open("parameters.yaml", "r") as stream:
            parameters = yaml.safe_load(stream)

        self.iou_thres = parameters["iou"]["correct_thres"]
        self.iou_correct_mul = parameters["iou"]["correct_point_multiplier"]
        self.iou_wrong_mul = parameters["iou"]["wrong_multiplier"]
        #print("IOU Hesaplayıcısı oluşturuldu. Doğru Thres, Doğru Çarpan, Yanlış Çarpan:", self.iou_thres, self.iou_correct_mul, self.iou_wrong_mul)

    def iou_calculate_one_frame(self, GT_coords, model_coords):
        """
        just input GT and models output
        returns max_point, current_point, positive_points, negative_points, mistakes extra detections, mistakes lost detections
        """
        if GT_coords.size != 0:
            gt_differences = np.ones(GT_coords.shape[0])
        else:
            gt_differences = np.array([])
        model_coords = np.array(model_coords)
        if not model_coords.size:
            return GT_coords.shape[0] * self.iou_correct_mul, 0, 0, 0, np.array([]), gt_differences

        negative_scores = np.ones(model_coords.shape[0])

        negative_scores = negative_scores * self.iou_wrong_mul
        iou_results = []
        
        for i, GT in enumerate(GT_coords):
            result_one_object = self.iou_calculate_one_ground_truth(GT, model_coords, self.iou_thres)
            for result in result_one_object:
                negative_scores[int(result[1])] = 0
            if len(result_one_object):
                iou_results.append(max(result_one_object)[0] * self.iou_correct_mul)
                gt_differences[i] = 0
        
        total_iou = sum(iou_results) + np.sum(negative_scores)

        return GT_coords.shape[0] * self.iou_correct_mul, total_iou, sum(iou_results), np.sum(negative_scores), negative_scores, gt_differences
    
    @staticmethod
    @jit(nopython=True)
    def iou_calculate_one_ground_truth(GT, models_coords, iou_thres):
        matched = []
        
        for i, coord in enumerate(models_coords):

            if GT[4] != coord[5]:
                continue

            inter_x_diff = (min(GT[2], coord[2]) - max(GT[0], coord[0]))
            inter_y_diff = (min(GT[3], coord[3]) - max(GT[1], coord[1]))
            
            if inter_x_diff <= 0 or inter_y_diff <= 0:
                continue
            
            inter_area = inter_x_diff * inter_y_diff
            GT_area = (GT[2] - GT[0]) * (GT[3] - GT[1])
            model_area = (coord[2] - coord[0]) * (coord[3] - coord[1])
            iou = inter_area / (GT_area + model_area - inter_area)
            
            if iou >= iou_thres:
                matched.append([iou, i])
        
        return matched

class iou_optimizer():
    def __init__(self):
        with open("parameters.yaml", "r") as stream:
            parameters = yaml.safe_load(stream)

        self.iou_thres = parameters["iou"]["correct_thres"]
        self.iou_correct_mul = parameters["iou"]["correct_point_multiplier"]
        self.iou_wrong_mul = parameters["iou"]["wrong_multiplier"]
        self.model_name = parameters["yolov5"]["weights"]
        self.GT = []
        self.output = []
        self.iou_calculate = iou_calculator()
        self.plot_names = []
        np.seterr(invalid='ignore')

    def add_sample(self, GT, output):
        self.GT.append(GT)
        self.output.append(output)
    
    def clear_sample(self):
        self.GT = []
        self.output = []

    def annot_max(self, x,y):
        xmax = x[np.argmax(y)]
        ymax = y.max()
        text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data',textcoords="axes fraction",
                arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        plt.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

    def create_histogram(self, flow_name, inform = True): #!iyileştir
        """
        En küçük conf değerinden 1'e kadar olan değerler arasında iou puanı histogramı oluşturur
        """
        great_totals = []
        max_total = 0
        for output, GT in zip(self.output, self.GT):
            output = np.array(output)
            totals = []
            for conf_candidate in range(100):
                conf_candidate = conf_candidate / 100
                if output.shape[0] > 0:
                    output_candidate = output[output[:,4] > conf_candidate]
                else:
                    output_candidate = []
                max, total, plus, minus, _, _= self.iou_calculate.iou_calculate_one_frame(GT, output_candidate)
                totals.append(total)
                if conf_candidate == 0:
                    max_total += max
            great_totals.append(totals)
            if inform:
                print("Histogram Creation Progress:", len(great_totals)-1, "/", len(self.GT), end = "\r")
        histogram = np.sum(great_totals, axis = 0)
        histogram = histogram / max_total
        max_value = np.max(histogram)
        if inform:
            print(max_value, "is the iou max value at", np.argmax(histogram)/100, "conf value")
        X = np.arange(0, 100) / 100
        return np.nan_to_num(histogram)
        
