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
from util.yolor import Yolor
from yolor.utils.general import set_logging, increment_path
from yolor.utils.plots import plot_one_box
from yolor.utils.torch_utils import select_device, time_synchronized

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
                'StartTimestamp': datetime(2022,12,8,0,45,00), #jp_test4
                'EndTimestamp': datetime(2022,12,8,1,00,00)
                # 'StartTimestamp': datetime(2022,12,7,16,10,00), #jp_test4
                # 'EndTimestamp': datetime(2022,12,7,16,23,00)
                # 'StartTimestamp': datetime(2022,12,17,5,45,00),  #jp_test2
                # 'EndTimestamp': datetime(2022,12,17,5,55,00)
                # 'StartTimestamp': datetime(2022,12,17,12,35,00),  #jp_test2
                # 'EndTimestamp': datetime(2022,12,17,12,45,00)
                # 'StartTimestamp': datetime(2022,12,18,2,40,00),  #jp_test2
                # 'EndTimestamp': datetime(2022,12,18,2,50,00)
                # 'StartTimestamp': datetime(2022,12,8,12,10,00),  #jp_test6
                # 'EndTimestamp': datetime(2022,12,8,12,20,00)
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

def update_tracks(work_area_index, workspaces, tracker, im0, width, height, ignored_classes, suspended_threshold_hatch, suspended_threshold_wharf, wharf, frame_id, fps, no_action):
    if (no_action and not wharf) and work_area_index != -1:
        return [], [], [],[],[]

    boxes = []
    classes = []
    old_classes = []
    detections = []
    ids = []
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

    # Consider a cargo traveling a constant distance in 0.5 seconds as valid cargo.
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
                        logger.debug(f'###########  valid cargo   {obj_id}')
                    else:
                        classes[i] = 'nested object' # Set as an arbitrary label, not a 'Suspended Lean Object'

    for i, box in enumerate(boxes):
        class_name = classes[i]
        detection = detections[i]
        track_id = ids[i]
        if (not class_name in ignored_classes) and class_name != 'nested object':
            xyxy = xywh2xyxy(box)
            original_label = f'{detection} #{track_id}'
            label = original_label
            # plot_one_box(xyxy, im0, label=label,color=get_color_for(original_label), line_thickness=3)
    return boxes, classes, old_classes, ids

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

def get_detection_frame_yolor(frame, engine):
    s_time = time.time()
    imgs_array = []
    imgsz = opt.img_size
    
    det = engine.inference_frame(frame, img_size = imgsz, auto_size = 64)
    e_time = time.time()
    # print(f'Detection is done for frames in {(e_time - s_time)} seconds!')
    return det   

def get_deepsort_tracker(max_cosine_distance, nn_budget):
    # initialize deep sort
    # model_filename = "ViT-B/16"
    model_filename = constants.CLIP_MODEL_PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load(model_filename, device=device, jit=False)
    model.eval()
    encoder = gdet.create_box_encoder(model, transform, batch_size=1, device=device)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    
    tracker = Tracker(metric)
    return tracker, encoder

def detect(opt):
    # Initialize
    set_logging()      

    # Adding for KVS
    source = get_kiesis_url()
    logger.debug(source)
    opt.source=source
    # source, imgsz = opt.source, opt.img_size
    
    global inf_1, inf_2, inf_3, inf_4, inf_5
    ignored_classes = opt.ignored_classes
    for idx in range(len(ignored_classes)):
        ignored_classes[idx] = ignored_classes[idx].replace('_', ' ')
    
    """ initialize deepsort tracker """
    nms_max_overlap = opt.nms_max_overlap
    tracker, encoder = get_deepsort_tracker(opt.max_cosine_distance, opt.nn_budget)
    
    """ load yolor model here """
    engine = Yolor()
    global names
    names = engine.get_names()
    
    """ initialize segmentation model """
    workspace_detector = DetectWorkspace(opt.wharf)
    workspaces, workspace_contours = [], []

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                    exist_ok=opt.exist_ok))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # init VideoCapture
    # url = get_kiesis_url()
    cap = cv2.VideoCapture(source)
    vid_path, vid_writer = None, None
    if not cap.isOpened():
        logger.debug("Cannot open camera")
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

    suspended_threshold_hatch, suspended_threshold_wharf = 0, 0
    logger.debug('Safety zone detection starts...')
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)  
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.debug("Can't receive frame (stream end?). Exiting ...")
            break

        c0 = time.time()
        if frame_count == 0:
            distance_tracker = DistanceTracker(frame, opt.source, height, width, fps, ignored_classes, opt.danger_zone_width_threshold, opt.danger_zone_height_threshold, workspaces, workspace_contours, opt.wharf, save_dir, opt.save_result)

        if (frame_count / fps) % 300 == 0 or len(workspaces) == 0: # detect work area once every 5 mins
            cargo_tracker.clear()
            detection = get_detection_frame_yolor(frame, engine)
            if len(detection) == 0:
                frame_count = frame_count+1
                if frame_count == frames:
                    break
                if cv2.waitKey(1) == ord('q'):
                    break
                continue

            workspaces, center_points, workspace_contours = workspace_detector.segment_workspace(frame)
            if len(workspaces) == 0:
                frame_count = frame_count+1
                if frame_count == frames:
                    break
                if cv2.waitKey(1) == ord('q'):
                    break
                continue

            if len(workspaces) == 1:
                logger.debug(f'*********************   only 1 work area')
                work_area_index = 0
                if opt.wharf:
                    _,_,_,wharf_ground_height = cv2.boundingRect(workspaces[0])
                    
                distance_tracker.update_workarea(workspaces[work_area_index])
                distance_tracker.calibrate_reference_area('')
                suspended_threshold_hatch, suspended_threshold_wharf = distance_tracker.get_suspended_threshold()

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
                suspended_threshold_hatch, suspended_threshold_wharf = distance_tracker.get_suspended_threshold()

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
                bboxes, classes, old_classes, ids = update_tracks(work_area_index, workspaces, tracker, frame, width, height, ignored_classes, suspended_threshold_hatch, suspended_threshold_wharf, opt.wharf, frame_count, fps, not hasattr(distance_tracker, 'distance_w'))
                
                # track unloading cargo and detect main work area from candidates
                result, work_area_index = cargo_tracker.track(work_area_index, ids, classes, bboxes, center_points, workspaces, workspace_contours)
                if result:
                    distance_tracker.update_workarea(workspaces[work_area_index])
                    distance_tracker.calibrate_reference_area('')
                    suspended_threshold_hatch, suspended_threshold_wharf = distance_tracker.get_suspended_threshold()

                c4 = time.time()
                inf_4 = int((c4-c3) *1000)

                # update distance tracker
                distance_tracker.calculate_distance(work_area_index, bboxes, classes, old_classes, frame, frame_count,ids, opt.thr_f_h, wharf_landing_Y, hatch_reference, db_manager)
                # Print time (inference + NMS)
                c5 = time.time()
                inf_5 = int((c5-c4) *1000)
            
        frame_count = frame_count+1
        if frame_count == frames:
            break

        elapsed_time = int((time.time()-c0)*1000)
        print(f'\t\t\t elapsed time {elapsed_time}: {inf_1}, {inf_2}, {inf_3}, {inf_4}, {inf_5}')
        
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
    logger.debug(opt)
    if get_camera_area() == 'hatch':
        opt.wharf = False
    else:
        opt.wharf = True

    with torch.no_grad():
        detect(opt)
