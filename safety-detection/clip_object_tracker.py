import sys
sys.path.insert(0, './yolor')
import argparse
import time
from datetime import datetime
from pathlib import Path
from turtle import distance
import glob
import os
import requests

import clip
import time
import cv2
import torch
import numpy as np
import math
import boto3
from loguru import logger

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_clip_detections as gdet

from util.segment import DetectWorkspace
from util.distance import DistanceTracker
from util.cargo_tracker import CargoTracker
from util.db_manager import DBManager
import util.threads as threads

import constants
from utils_.yolor import Yolor
# from utils_.yolov5 import Yolov5, Yolov5_IV
if constants.USE_YOLOR_MODEL:
    from yolor.utils.datasets import LoadImages
    from yolor.utils.general import set_logging, increment_path
    from yolor.utils.plots import plot_one_box
    from yolor.utils.torch_utils import select_device, time_synchronized
else:
    from fastai.vision.all import *
    from yolov5.utils.datasets import LoadImages
    from yolov5.utils.general import set_logging, increment_path
    from yolov5.utils.plots import plot_one_box
    from yolov5.utils.torch_utils import select_device, time_synchronized

classes = []
names = []
all_cargos =dict()
inf_1, inf_2, inf_3, inf_4, inf_5 = 0, 0, 0, 0, 0

def xyxy2xywh(xyxy):
    x = xyxy[0]
    y = xyxy[1]
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    return (x,y,w,h)

def xywh2xyxy(xywh):
    x1 = xywh[0]
    y1 = xywh[1]
    x2 = xywh[2] + xywh[0]
    y2 = xywh[3] + xywh[1]
    return (x1,y1,x2,y2)

def is_inside(point, box):
        x, y, w, h = box
        return point[0] >= x and point[0] <= x+w and point[1] >= y and point[1] <= y+h

def get_kiesis_url(live=True):
    live = True if os.environ.get("live") == "1" \
            else False
    STREAM_NAME = os.environ.get('kinesis_url')
    kvs = boto3.client("kinesisvideo", )
    # Grab the endpoint from GetDataEndpoint
    endpoint = kvs.get_data_endpoint(
        APIName="GET_HLS_STREAMING_SESSION_URL",
        StreamName=STREAM_NAME
        )['DataEndpoint']
    
    kvam = boto3.client("kinesis-video-archived-media", endpoint_url=endpoint)
    if live:
        url = kvam.get_hls_streaming_session_url(
            StreamName=STREAM_NAME,
            #PlaybackMode="ON_DEMAND",
            PlaybackMode="LIVE",
            Expires = int(12*3600)
            )['HLSStreamingSessionURL']
    else: 
        url = kvam.get_hls_streaming_session_url(
            StreamName=STREAM_NAME,
            #PlaybackMode="ON_DEMAND",
            PlaybackMode="LIVE_REPLAY",
            HLSFragmentSelector={
            'FragmentSelectorType': 'SERVER_TIMESTAMP',
            'TimestampRange': {
                'StartTimestamp': datetime(2022,12,7,17,10,0),
                'EndTimestamp': datetime(2022,12,7,17,30)
                }
            },
            Expires = int(12*3600)
            )['HLSStreamingSessionURL']

    return url

def get_camera_area():

    base_url = "https://jp-safety.groundup.ai/api/"
    url = "https://jp-safety.groundup.ai/api/kinesis-stream"
    # Issue jwt token first 
    auth_payload = "{\n    \"email\":\"admin@groundup.ai\",\n    \"password\":\"Password1$\"\n}"
    headers = {
    'Content-Type': 'application/json'
    }
    token_response = requests.request("POST", base_url+"auth/login", headers=headers, data=auth_payload)
    payload={}
    headers = {
        'Authorization': 'Bearer ' + str(token_response.json()['token'])
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    all_cams_meta = response.json()
    cam_area = 'wharf'
    stream_name = os.environ.get('kinesis_url')
    for each_cam in all_cams_meta:
        # logger.debug(each_cam)
        if each_cam['name'] == stream_name:
            area = each_cam['area']
            if area is not None:
                if 'hatch' in area.lower().split(' '):
                    cam_area = 'hatch'
                return cam_area
        
    return cam_area

def update_tracks(work_area_index, workspaces, tracker, im0, width, height, ignored_classes, suspended_threshold_hatch, suspended_threshold_wharf, suspended_threshold_wharf_side, angle, distance_check, wharf, frame_id, fps, no_action, no_nested):
    if (no_action and not wharf) and work_area_index != -1:
        return [], [], [],[],[]
    
    max_distances = {
        'Forklift': constants.MAX_DISTANCE_FOR_FORKLIFT,
        'Suspended Lean Object': constants.MAX_DISTANCE_FOR_SUSPENDED_LEAN_OBJECT,
        'chain': constants.MAX_DISTANCE_FOR_CHAIN,
        'People': constants.MAX_DISTANCE_FOR_PEOPLE,
        'Human Carrier': constants.MAX_DISTANCE_FOR_HUMAN_CARRIER,
    }

    human_height_coefficient = constants.HUMAN_HEIGHT_COEFFICIENT
    forklift_height_coefficient = constants.FORKLIFT_HEIGHT_COEFFICIENT
    human_carrier_height_coefficient = constants.HUMAN_CARRIER_HEIGHT_COEFFICIENT

    boxes = []
    classes = []
    old_classes = []
    detections = []
    ids = []
    heights = []
    areas = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        xyxy = track.to_tlbr()
        xywh = xyxy2xywh(xyxy)
        class_num = track.class_num
        class_name = names[int(class_num)]
        if class_name in ignored_classes:
            continue
        boxes.append(xywh)
        classes.append(class_name)
        old_classes.append(class_name)
        detections.append(class_name)
        ids.append(track.track_id)

    suspended_threshold = suspended_threshold_wharf if wharf else suspended_threshold_hatch
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        if classes[i] in ['Container', 'Small Pipe', 'Large Pipe', 'Wooden Board', 'Iron Rake', 'Wood', 'Coil', 'Steel Plate']:
            classes[i] = 'Suspended Lean Object'

        if work_area_index == -1:
            continue
            
        if classes[i] in ['Suspended Lean Object', 'People', 'chain'] + ignored_classes:
            continue
            
        if not wharf:
            if y + h <= suspended_threshold:
                if h / height >= 0.08:
                    classes[i] = 'Suspended Lean Object'
        else: # consider only "Forklift","HumanCarrier" outside the ground as 'Suspended Lean Object' so that they can take part in danager area detection
            inside = cv2.pointPolygonTest(workspaces[0], [x + w/2, y + h], False)
            if inside < 0 and h / height >= 0.08:
                classes[i] = 'Suspended Lean Object'
    
    for i in range(len(boxes)):
        class_name = classes[i]
        xywh = boxes[i]
        if class_name in ignored_classes:
            continue
        if class_name in ['People', 'Human Carrier', 'Forklift']:
            if xywh[3]/height > 0:
                heights.append(xywh[3]/height)
            else:
                heights.append(1)
            areas.append(-1)
        else:
            heights.append(-1)
            if (xywh[2] * xywh[3]) / (height * width) > 0:
                areas.append( (xywh[2] * xywh[3]) / (height * width) )
            else:
                areas.append(1)

    # only for converting chain to suspended lean object. if the chain is ignored, no need for it!
    if not 'Suspended Lean Object' in classes and not 'chain' in ignored_classes:
        for i, cls in enumerate(classes):
            if cls == 'chain':
                x, y, w, h = boxes[i]
                bottom1 = (x, y+h)
                bottom2 = (x+w, y+h)
                if y <= suspended_threshold:
                    not_inside = True
                    for j, box in enumerate(boxes):
                        if classes[j] != 'People' and i != j:
                            if is_inside(bottom1, box) or is_inside(bottom2, box):
                                # print(bottom1, bottom2, box)
                                not_inside = False
                    if not_inside:
                        classes[i] = 'Suspended Lean Object'

    try:
        if no_nested:
            nested_ids = []
            for i, box_i in enumerate(boxes):
                # check whether it is in a forklift or a human carrier
                for j, box_j in enumerate(boxes):
                    if detections[j] == 'Forklift' and detections[i] == 'Forklift' and i != j:
                        p1 = (box_i[0] , box_i[1] )
                        p2 = (box_i[0] , box_i[1]+box_i[3] )
                        p3 = (box_i[0] + box_i[2] , box_i[1] )
                        p4 = (box_i[0] + box_i[2] , box_i[1]+box_i[3] )
                        inside_bbox = is_inside(p1, box_j) and is_inside(p2, box_j) and is_inside(p3, box_j) and is_inside(p4, box_j)
                        if inside_bbox:
                            nested_ids.append(i)
            # print([(ids[idx]) for idx in nested_ids])
            for nested_id in list(set(nested_ids)):
                classes[nested_id] = 'nested object'
    except:
        print('No nested error has been handled!')

    # if there is no people, we have not reference measurement and we cannot do distance estimation
    if not 'People' in classes or not wharf:
        distance_check = False

    distance_estimations = []
    if distance_check:
        max_people_height = -1.0
        i = 0
        while i < len(classes):
            if classes[i] == 'People' and heights[i] > max_people_height:
                max_people_height = heights[i]
            i+=1
        first_object_distance = human_height_coefficient / max_people_height
        first_objects = {cls: -1.0 for cls in classes if not cls in ['People', 'Human Carrier', 'Forklift']}
        for i, cls in enumerate(classes):
            if not cls in ['People', 'Human Carrier', 'Forklift']:
                first_objects[cls] = max(first_objects[cls], areas[i])
        for i, cls in enumerate(classes):
            if not cls in ['People', 'Human Carrier', 'Forklift']:
                reference_area = first_objects[cls]
                reference_distance = first_object_distance
                object_area = areas[i]
                object_distance = np.sqrt(np.square(reference_distance) * reference_area / object_area)
                distance_estimations.append(object_distance)
            elif cls == 'People':
                object_distance = human_height_coefficient / heights[i]
                distance_estimations.append(object_distance)
            elif cls == 'Human Carrier':
                object_distance = human_carrier_height_coefficient / heights[i]
                distance_estimations.append(object_distance)
                # print(object_distance, heights[i])
            elif cls == 'Forklift':
                object_distance = forklift_height_coefficient / heights[i]
                distance_estimations.append(object_distance)
            #far_check[i] = object_distance < max_distances[cls]
            # print(i, distance_estimations[-1], classes[i], detections[i])

        for i, box_i in enumerate(boxes):
            try:
                if classes[i] == 'nested object':
                    continue
                # check whether it is in a forklift or a human carrier
                for j, box_j in enumerate(boxes):
                    if classes[j] in ['Forklift', 'Human Carrier'] and classes[i] == 'People':
                        p1 = (box_i[0] , box_i[1] )
                        p2 = (box_i[0] , box_i[1]+box_i[3] )
                        p3 = (box_i[0] + box_i[2] , box_i[1] )
                        p4 = (box_i[0] + box_i[2] , box_i[1]+box_i[3] )
                        inside_bbox = is_inside(p1, box_j) and is_inside(p2, box_j) and is_inside(p3, box_j) and is_inside(p4, box_j)
                        if inside_bbox:
                            distance_estimations[i] = distance_estimations[j]
                if distance_estimations[i] > max_distances[classes[i]] or (classes[i] != 'People' and (box_i[3] * box_i[2]) / (height * width) < 0.0025):
                    classes[i] = 'far object'
            except:
                continue
    

    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        thr_frames = int(fps * 0.5)
        if classes[i] in ['Suspended Lean Object']:
            obj_id = ids[i]
            mid_point = [x + w//2, y + h//2]
            if obj_id not in all_cargos.keys():
                cargo_dict = {'first_frame_id':frame_id, 'first_pos':mid_point, 'valid': False}
                all_cargos[obj_id] = cargo_dict

            diff_frames = frame_id - all_cargos[obj_id]['first_frame_id']
            if diff_frames < thr_frames:
                classes[i] = 'nested object' # Set as an arbitrary label, not a 'Suspended Lean Object'
            else:
                if all_cargos[obj_id]['valid'] == False:
                    move_distance = math.dist(mid_point, all_cargos[obj_id]['first_pos'])
                    if move_distance > int(height / 16):
                        all_cargos[obj_id]['valid'] = True
                        print(f'###########  valid cargo   {obj_id}')
                    else:
                        classes[i] = 'nested object' # Set as an arbitrary label, not a 'Suspended Lean Object'

    for i, box in enumerate(boxes):
        class_name = classes[i]
        detection = detections[i]
        track_id = ids[i]
        cur_distance = int(distance_estimations[i]) if distance_check else -1
        if (not class_name in ignored_classes) and class_name !='far object' and class_name != 'nested object':
            xyxy = xywh2xyxy(box)
            original_label = f'{detection} #{track_id}'
            label = f'{original_label} {cur_distance} CM' if distance_check else original_label
            # plot_one_box(xyxy, im0, label=label,color=get_color_for(original_label), line_thickness=3)
    return boxes, classes, old_classes, distance_estimations,ids

def get_color_for(class_num):
    colors = [
        "#4892EA",
        "#00EEC3",
        "#FE4EF0",
        "#F4004E",
        "#FA7200",
        "#EEEE17",
        "#90FF00",
        "#78C1D2",
        "#8C29FF"
    ]

    num = hash(class_num) # may actually be a number or a string
    hex = colors[num%len(colors)]

    # adapted from https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    rgb = tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    return rgb

def get_all_detections_yolov5(vid_path, engine, budget):
    print('Detection YOLOv5 starts...')
    s_time = time.time()
    imgs_array = []
    dets = []
    cap = cv2.VideoCapture(vid_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = frames if budget == -1 else min(frames, int(fps*60*budget))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_count = 0
    batch_size = constants.DETECTION_BATCH_SIZE
    while total_count < frames:
        ret_val, img = cap.read()
        if ret_val and type(img) != type(None):
            imgs_array.append(img)
        if len(imgs_array) == batch_size:
            try:
                dets += engine.batch_inference(imgs_array)
            except:
                print('Batch inference error!')
            imgs_array = []
        total_count += 1
        """if total_count % 1000 == 0:
            print(total_count)
            time.sleep(1)"""
    if len(imgs_array) > 0:
        try:
            dets += engine.batch_inference(imgs_array)
        except:
            print('Batch inference error!')
        imgs_array = []
    cap.release()
    e_time = time.time()
    print(f'Detection is done for {frames} frames in {(e_time - s_time)} seconds!')
    return dets, height, width, fps 

def get_all_detections_yolor(vid_path, dataset, engine, budget):
    print('Detection YOLOR starts...')
    s_time = time.time()
    imgs_array = []
    dets = []
    cap = cv2.VideoCapture(vid_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = frames if budget == -1 else min(frames, int(fps*60*budget))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_count = 0
    batch_size = constants.DETECTION_BATCH_SIZE
    dets = engine.inference(dataset)
    cap.release()
    e_time = time.time()
    print(f'Detection is done for {frames} frames in {(e_time - s_time)} seconds!')
    return dets, height, width, fps 

def get_detection_frame_yolor(frame, engine):
    s_time = time.time()
    imgs_array = []
    imgsz = opt.img_size
    
    det = engine.inference_frame(frame, img_size = imgsz, auto_size = 64)
    e_time = time.time()
    # print(f'Detection is done for frames in {(e_time - s_time)} seconds!')
    return det   

def detect(opt):
    # Adding for KVS
    source = get_kiesis_url()
    logger.debug(source)
    opt.source=source
    # Support enabled for wharf in uat only
    if os.environ["env"] == "prod" and get_camera_area() == 'wharf':
        # At this stage we need to exit as it is not able to work for wharf
        logger.info("At prod wharf currently model is not supported")
        sys.exit()

    global inf_1, inf_2, inf_3, inf_4, inf_5
    t0 = time_synchronized()
    ignored_classes = opt.ignored_classes
    for idx in range(len(ignored_classes)):
        ignored_classes[idx] = ignored_classes[idx].replace('_', ' ')
    nms_max_overlap = opt.nms_max_overlap
    max_cosine_distance = opt.max_cosine_distance
    nn_budget = opt.nn_budget

    # initialize deep sort
    model_filename = "ViT-B/16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load(model_filename, device=device, jit=False)
    model.eval()
    encoder = gdet.create_box_encoder(model, transform, batch_size=1, device=device)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    
    # load yolov5 model here
      
    engine = Yolor() if constants.USE_YOLOR_MODEL else Yolov5_IV() if constants.USE_IV_MODEL else Yolov5()
    global names
    names = engine.get_names()
    workspace_detector = DetectWorkspace(opt.wharf)

    # initialize tracker
    tracker = Tracker(metric)

    # source, imgsz = opt.source, opt.img_size

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                    exist_ok=opt.exist_ok))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()      

    # init VideoCapture
    # url = get_kiesis_url()
    cap = cv2.VideoCapture(source)
    vid_path, vid_writer = None, None
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = 12
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = frames if opt.budget == -1 else min(frames, int(fps*60*opt.budget))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    frame_count = 0
    distance_tracker = None
    calibrated_frames = 0

    # init DBManager
    db_manager = DBManager(source, width, height, fps, save_dir)

    # initialize cargo tracker
    work_area_index = -1
    cargo_tracker = CargoTracker(opt.wharf, fps, height, width)
    wharf_landing_Y = -1
    wharf_ground_height = 0
    hatch_reference = []
    cargo_ids = []

    suspended_threshold_hatch, suspended_threshold_wharf, suspended_threshold_wharf_side = 0, 0, 0
    workspaces, workspace_contours = [], []
    print('Safety zone detection starts...')
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)  
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        c0 = time.time()
        if frame_count == 0:
            distance_tracker = DistanceTracker(frame, opt.source, height, width, fps, ignored_classes, opt.danger_zone_width_threshold, opt.danger_zone_height_threshold, workspaces, workspace_contours, opt.wharf, opt.angle, save_dir, opt.save_result)

        if (frame_count / fps) % 300 == 0 or len(workspaces) == 0: # detect work area once every 5 mins
            cargo_tracker.clear()
            detection = get_detection_frame_yolor(frame, engine)
            if len(detection) == 0:
                frame_count = frame_count+1
                if frame_count == frames:
                    break
                
                # cv2.imshow("output", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                continue
            workspaces, center_points, workspace_contours = workspace_detector.segment_workspace(frame)
            if len(workspaces) == 0:
                frame_count = frame_count+1
                if frame_count == frames:
                    break
                
                # cv2.imshow("output", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                continue

            if len(workspaces) == 1:
                print(f'*********************   only 1 work area')
                work_area_index = 0
                # if not opt.wharf:
                #     cargo_tracker.set_step(3)
                if opt.wharf:
                    _,_,_,wharf_ground_height = cv2.boundingRect(workspaces[0])
                    
                distance_tracker.update_workarea(workspaces[work_area_index])
                distance_tracker.calibrate_reference_area('')
                suspended_threshold_hatch, suspended_threshold_wharf, suspended_threshold_wharf_side = distance_tracker.get_suspended_threshold()

            distance_tracker.update_edgepoints(workspace_contours)
        
        if opt.wharf:
            if cargo_tracker.get_wharf_landing_Y() > 0:
                wharf_landing_Y = cargo_tracker.get_wharf_landing_Y()
            # if wharf_landing_Y > 0:
            #     pt1 = (0, wharf_landing_Y)
            #     pt2 = (width, wharf_landing_Y)
                # frame = cv2.line(frame, pt1, pt2, (0, 255, 0), 5)
        else:
            hatch_reference = cargo_tracker.get_hatch_reference()
            # if len(hatch_reference) != 0:
            #     frame = cv2.circle(frame, hatch_reference[0], 20, (0, 255, 0), 20)

        detection = get_detection_frame_yolor(frame, engine)
        c1 = time.time()
        inf_1 = int((c1-c0) *1000)

        if work_area_index != -1 and not opt.wharf:
            if calibrated_frames < 10:
                if calibrated_frames == 0:
                    distance_tracker.clear_calibration_count()
                calib_bboxes = []
                for bbox in detection:
                    if bbox[-1] == names.index('People'): 
                        calib_bboxes.append(bbox.detach().cpu())
                if len(calib_bboxes) > 0:
                    calibrated_frames += 1
                    distance_tracker.calibrate_lengths(calib_bboxes)
            else:
                calibrated_frames = (calibrated_frames + 1) % 30

        if wharf_landing_Y != -1 and opt.wharf:
            calib_bboxes = []
            for bbox in detection:
                if bbox[-1] == names.index('People'): 
                    calib_bboxes.append(bbox.detach().cpu())
            if len(calib_bboxes) > 0:
                frame = distance_tracker.calibrate_person_height_wharf(frame, calib_bboxes, wharf_landing_Y, wharf_ground_height)

        pred = detection.unsqueeze(0)
        c2 = time.time()
        inf_2 = int((c2-c1) *1000)
        
        if len(pred) == 0: # detect main work area from candidates
            result, work_area_index = cargo_tracker.track_no_detection_case(work_area_index, workspace_contours)
            if result:
                distance_tracker.update_workarea(workspaces[work_area_index])
                distance_tracker.calibrate_reference_area('')
                suspended_threshold_hatch, suspended_threshold_wharf, suspended_threshold_wharf_side = distance_tracker.get_suspended_threshold()

        cv2.drawContours(frame, workspace_contours, -1, (0, 0, 255), 5)
        
        for i, det in enumerate(pred): # detections per image
            bboxes = []
            classes = []
            if len(det):
                c3 = -1
                # Transform bboxes from tlbr to tlwh
                trans_bboxes = det[:, :4].clone()
                trans_bboxes[:, 2:] -= trans_bboxes[:, :2]
                bboxes = trans_bboxes[:, :4].cpu()
                confs = det[:, 4]
                class_nums = det[:, -1].cpu()
                classes = class_nums

                # encode yolo detections and feed to tracker
                features = encoder(frame, bboxes)
                detections = [Detection(bbox, conf, class_num, feature) for bbox, conf, class_num, feature in zip(
                    bboxes, confs, classes, features)]

                # run non-maxima supression
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                class_nums = np.array([d.class_num for d in detections])
                indices = preprocessing.non_max_suppression(
                    boxs, class_nums, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]
                c3 = time.time()
                inf_3 = int((c3-c2) *1000)
                
                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                # update tracks
                bboxes, classes, old_classes, distance_estimations,ids = update_tracks(work_area_index, workspaces, tracker, frame, width, height, ignored_classes, suspended_threshold_hatch, suspended_threshold_wharf, suspended_threshold_wharf_side, opt.angle, opt.distance_check, opt.wharf, frame_count, fps, not hasattr(distance_tracker, 'distance_w'), opt.no_nested)
                #print("length of bboxes {}".format(bboxes))
                #print("length of ids {}".format(len(ids)))
                
                # track unloading cargo and detect main work area from candidates
                result, work_area_index = cargo_tracker.track(work_area_index, ids, classes, bboxes, center_points, workspaces, workspace_contours)
                if result:
                    distance_tracker.update_workarea(workspaces[work_area_index])
                    distance_tracker.calibrate_reference_area('')
                    suspended_threshold_hatch, suspended_threshold_wharf, suspended_threshold_wharf_side = distance_tracker.get_suspended_threshold()

                c4 = time.time()
                inf_4 = int((c4-c3) *1000)

                # update distance tracker
                distance_tracker.calculate_distance(work_area_index, bboxes, classes, old_classes, distance_estimations, frame, frame_count,ids, opt.thr_f_h, wharf_landing_Y, hatch_reference, db_manager)
                # Print time (inference + NMS)
                c5 = time.time()
                inf_5 = int((c5-c4) *1000)
            
        frame_count = frame_count+1
        if frame_count == frames:
            break

        elapsed_time = int((time.time()-c0)*1000)
        # print(f'\t\t\t elapsed time {elapsed_time}: {inf_1}, {inf_2}, {inf_3}, {inf_4}, {inf_5}')
        # Display the resulting frame
        # cv2.imshow("output", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    threads.thread_manager.join_threads()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=float, default=-1.0)
    parser.add_argument('--source', required=True)
    parser.add_argument('--wharf', action='store_true')
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument('--angle', choices=['left', 'right'], default='left')
    parser.add_argument('--no_nested', action='store_true')
    parser.add_argument('--distance_check', action='store_true')
    parser.add_argument('--ignored_classes', nargs='+', default=['chain'])
    parser.add_argument('--danger_zone_width_threshold', type=float, default=400.0)
    parser.add_argument('--danger_zone_height_threshold', type=float, default=500.0)
    parser.add_argument('--img-size', type=int, default=1024,
                        help='inference size (pixels)')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--nms_max_overlap', type=float, default=1.0,
                        help='Non-maxima suppression threshold: Maximum detection overlap.')
    parser.add_argument('--max_cosine_distance', type=float, default=0.1,
                        help='Gating threshold for cosine distance metric (object appearance).')
    parser.add_argument('--nn_budget', type=int, default=None,
                        help='Maximum size of the appearance descriptors allery. If None, no budget is enforced.')

    parser.add_argument('--thr_f_h', type=float, default=0.9,
                        help='Threshold for the fall_from_height')

    opt = parser.parse_args()
    print(opt)
    if get_camera_area() == 'hatch':
        opt.wharf = False
    else:
        opt.wharf = True

    with torch.no_grad():
        detect(opt)
