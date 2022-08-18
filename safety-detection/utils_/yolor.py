import argparse
import os
import platform
import shutil
import time
from pathlib import Path
from tkinter.font import names

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import constants

from yolor.utils.google_utils import attempt_load
from yolor.utils.datasets import LoadStreams, LoadImages, letterbox
from yolor.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from yolor.utils.plots import plot_one_box
from yolor.utils.torch_utils import select_device, load_classifier, time_synchronized


from yolor.models.models import *
from yolor.utils.datasets import LoadImages, LoadStreams
from yolor.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty'' strings (such as last line)

class Yolor:
    def __init__(self):
        self.device = torch.device('cuda')
        self.names = load_classes(constants.YOLOR_CLASS_NAMES)
        self.cfg = constants.YOLOR_CONFIG
        self.img_size = constants.YOLOR_IMG_SIZE
        self.model = Darknet(self.cfg, self.img_size).cuda()
        self.model.load_state_dict(torch.load(constants.DETECTION_MODEL_PATH, map_location=self.device)['model'])
        self.conf_thres = constants.DETECTION_CONF_THRESHOLD
        self.iou_thres = constants.YOLOR_IOU_THRESHOLD
        self.model.to(self.device).eval()
        half = self.device.type != 'cpu'
        if half:
            self.model.half()  # to FP16
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = self.model(img.half() if half else img) if self.device.type != 'cpu' else None  # run once

    def inference(self, dataset):
        half = self.device.type != 'cpu'

        predictions = []
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = self.model(img, augment=False)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)

            height, width = img.shape[:2]
            predictions_per_image = []
            s = ''
            for i, det in enumerate(pred):
                print(len(det))
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.names[int(c)])
                
                boxes = np.array([[box[0], box[1], box[2], box[3]] for box in det[:, 0:4].cpu()])
                scores = det[:, 4].cpu()
                labels = det[:, 5].cpu()
                

                if len(boxes) > 0:
                    if width>=height:
                        padding = (width-height)/2
                        box_wo_pad = boxes - padding * np.array([0,1,0,-1])
                    else:
                        padding = (height-width)/2
                        box_wo_pad = boxes - padding * np.array([1,0,-1,0])
                else:
                    box_wo_pad = boxes

                box_wo_pad = box_wo_pad.astype(int)

                print(f'{s}  Done.')
                for i, box in enumerate(box_wo_pad):
                    predictions_per_image.append(np.array([box[0], box[1], box[2], box[3], scores[i], labels[i]]))
                    label_index = labels[i].detach().numpy()

            predictions.append(torch.from_numpy(np.array(predictions_per_image)))

        return predictions
                
    def inference_frame(self, frame, img_size = 640, auto_size=32):
        half = self.device.type != 'cpu'
        
        img = letterbox(frame, new_shape=img_size, auto_size=auto_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)

        height, width = img.shape[:2]
        predictions_per_image = []
        s = ''
        for i, det in enumerate(pred):
            # print(len(det))
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])
            
            boxes = np.array([[box[0], box[1], box[2], box[3]] for box in det[:, 0:4].cpu()])
            scores = det[:, 4].cpu()
            labels = det[:, 5].cpu()
            

            if len(boxes) > 0:
                if width>=height:
                    padding = (width-height)/2
                    box_wo_pad = boxes - padding * np.array([0,1,0,-1])
                else:
                    padding = (height-width)/2
                    box_wo_pad = boxes - padding * np.array([1,0,-1,0])
            else:
                box_wo_pad = boxes

            box_wo_pad = box_wo_pad.astype(int)

            print(f'{s}  Done.')
            for i, box in enumerate(box_wo_pad):
                predictions_per_image.append(np.array([box[0], box[1], box[2], box[3], scores[i], labels[i]]))
                label_index = labels[i].detach().numpy()

        return (torch.from_numpy(np.array(predictions_per_image)))


    def get_names(self):
        return self.names