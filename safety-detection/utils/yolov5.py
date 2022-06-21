from models.experimental import attempt_load
from utils.general import non_max_suppression

from icevision.all import *
import gc
import torch
from pytorch_lightning.loggers import WandbLogger
from icevision.models import *
from PIL import Image
import numpy as np
import time
import constants

class Yolov5:
    def __init__(self):
        
        checkpoint_and_model = model_from_checkpoint(constants.DETECTION_MODEL_PATH)
        self.classes = checkpoint_and_model["class_map"].get_classes()[1:]
        print(self.classes)
        self.img_size = checkpoint_and_model["img_size"]
        self.model = torch.hub.load('ultralytics/yolov5', f'yolov5l', classes=len(self.classes), pretrained=False)
        self.model.model = checkpoint_and_model["model"]
        self.conf_thres = constants.DETECTION_CONF_THRESHOLD

    def parse_pred_dict(self, pred_dict):
        one_img_list=[]
        for i in range(len(pred_dict['detection']['bboxes'])):

            #print(i)
            xmin=pred_dict['detection']['bboxes'][i].xmin
            ymin=pred_dict['detection']['bboxes'][i].ymin
            xmax=pred_dict['detection']['bboxes'][i].xmax
            ymax=pred_dict['detection']['bboxes'][i].ymax
            label=pred_dict['detection']['label_ids'][i]
            score=pred_dict['detection']['scores'][i]

            b=[xmin,ymin,xmax,ymax,score,label]

            one_img_list.append(b)
        npa = np.asarray(one_img_list, dtype=np.float32)
        return npa

    def infer(self, img):
        if type(img) == str:
            img = PIL.Image.open(img)
            img = img.convert('RGB')
        else:
            img = Image.fromarray(img)
        pred_dict  = self.model_type.end2end_detect(img, self.valid_tfms, self.loaded_model, class_map=self.class_map, detection_threshold=self.conf_thres)
        return np.expand_dims(self.parse_pred_dict(pred_dict),0)

    def warmup(self, img):
        self.model(img)

    def batch_inference(self, ims):
        preds = self.model(ims, size=self.img_size)
        all_dets = [det.xyxy[0] for det in preds.tolist()]
        return all_dets
    

    def get_names(self):
        return self.classes

class Yolov5_IV:
    def __init__(self):
        device=torch.device('cuda')
        checkpoint_and_model = model_from_checkpoint(constants.DETECTION_MODEL_PATH)
        self.model_type = checkpoint_and_model["model_type"]
        backbone = checkpoint_and_model["backbone"]
        self.class_map = checkpoint_and_model["class_map"]
        img_size = checkpoint_and_model["img_size"]
        self.img_size = img_size
        self.loaded_model = checkpoint_and_model["model"]
        self.valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(img_size), tfms.A.Normalize()])
        self.classes = self.class_map.get_classes()
        print(len(self.classes))
        self.conf_thres = constants.DETECTION_CONF_THRESHOLD

    def parse_pred_dict(self, pred_dict):
        one_img_list=[]
        for i in range(len(pred_dict['detection']['bboxes'])):
            xmin=pred_dict['detection']['bboxes'][i].xmin
            ymin=pred_dict['detection']['bboxes'][i].ymin
            xmax=pred_dict['detection']['bboxes'][i].xmax
            ymax=pred_dict['detection']['bboxes'][i].ymax
            label=pred_dict['detection']['label_ids'][i]
            score=pred_dict['detection']['scores'][i]

            b=[xmin,ymin,xmax,ymax,score,label]

            one_img_list.append(b)
        npa = np.asarray(one_img_list, dtype=np.float32)
        return npa

    def infer(self, img):
        if type(img) == str:
            img = PIL.Image.open(img)
            img = img.convert('RGB')
        else:
            img = Image.fromarray(img)
        pred_dict  = self.model_type.end2end_detect(img, self.valid_tfms, self.loaded_model, class_map=self.class_map, detection_threshold=self.conf_thres)
        return np.expand_dims(self.parse_pred_dict(pred_dict),0)

    def batch_inference(self, ims):
        height, width = ims[0].shape[:2]
        infer_ds = Dataset.from_images(ims, self.valid_tfms, class_map=self.class_map)

        # Batch Inference
        infer_dl = self.model_type.infer_dl(infer_ds, batch_size=constants.DETECTION_BATCH_SIZE, shuffle=False)
        preds = self.model_type.predict_from_dl(self.loaded_model, infer_dl, keep_images=False, detection_threshold=self.conf_thres)
        predictions = []
        for pred in preds:
            detections = pred.pred.detection
            boxes = np.array([[box.xmin, box.ymin, box.xmax, box.ymax] for box in detections.bboxes])
            labels = detections.labels
            scores = detections.scores
            class_map = detections.class_map
            box_original_size = boxes * (max(height, width)/self.img_size)
            if len(boxes) > 0:
                if width>=height:
                    padding = (width-height)/2
                    box_wo_pad = box_original_size - padding * np.array([0,1,0,1])
                else:
                    padding = (height-width)/2
                    box_wo_pad = box_original_size - padding * np.array([1,0,1,0])
            else:
                box_wo_pad = box_original_size
            box_wo_pad = box_wo_pad.astype(int)
            boxes_final = []
            for i, box in enumerate(box_wo_pad):
                boxes_final.append(np.array([box[0], box[1], box[2], box[3], scores[i], class_map.get_by_name(labels[i])]))
            predictions.append(torch.from_numpy(np.array(boxes_final)))
        return predictions
    

    def get_names(self):
        return self.classes