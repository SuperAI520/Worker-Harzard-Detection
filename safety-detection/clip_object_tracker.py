import argparse
import time
from pathlib import Path
from turtle import distance

import clip
import time
import cv2
import torch
import numpy as np

from utils.datasets import LoadImages
from utils.general import set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_clip_detections as gdet

from utils.yolov5 import Yolov5, Yolov5_IV
from distance import DistanceTracker
from fastai.vision.all import *
import constants

classes = []

names = []
inf_1, inf_2, inf_3, inf_4 = 0, 0, 0, 0 

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
"""
def update_tracks(tracker, im0, width, height, ignored_classes, suspended_threshold_hatch, suspended_threshold_wharf, distance_check, wharf):
    # if len(tracker.tracks):
    #     print("[Tracks]", len(tracker.tracks))

    max_distances = {
        'forklift': constants.MAX_DISTANCE_FOR_FORKLIFT,
        'suspended lean object': constants.MAX_DISTANCE_FOR_SUSPENDED_LEAN_OBJECT,
        'chain': constants.MAX_DISTANCE_FOR_CHAIN,
        'people': constants.MAX_DISTANCE_FOR_PEOPLE,
        'human carrier': constants.MAX_DISTANCE_FOR_HUMAN_CARRIER,
    }

    human_height_coefficient = constants.HUMAN_HEIGHT_COEFFICIENT
    forklift_height_coefficient = constants.FORKLIFT_HEIGHT_COEFFICIENT
    human_carrier_height_coefficient = constants.HUMAN_CARRIER_HEIGHT_COEFFICIENT

    boxes = []
    classes = []
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
        ids.append(track.track_id)
        if class_name in ['people', 'human carrier', 'forklift']:
            heights.append(xywh[3]/height)
            areas.append(-1)
        else:
            heights.append(-1)
            areas.append( (xywh[2] * xywh[3]) / (height * width) )

    #far_check = [True] * len(classes)

    # if there is no people, we have not reference measurement and we cannot do distance estimation
    if not 'people' in classes:
        distance_check = False

    distance_estimations = []
    if distance_check:
        max_people_height = -1.0
        i = 0
        while i < len(classes):
            if classes[i] == 'people' and heights[i] > max_people_height:
                max_people_height = heights[i]
            i+=1
        first_object_distance = human_height_coefficient / max_people_height
        first_objects = {cls: -1.0 for cls in classes if not cls in ['people', 'human carrier', 'forklift']}
        for i, cls in enumerate(classes):
            if not cls in ['people', 'human carrier', 'forklift']:
                first_objects[cls] = max(first_objects[cls], areas[i])
        for i, cls in enumerate(classes):
            if not cls in ['people', 'human carrier', 'forklift']:
                reference_area = first_objects[cls]
                reference_distance = first_object_distance
                object_area = areas[i]
                object_distance = np.sqrt(np.square(reference_distance) * reference_area / object_area)
                distance_estimations.append(object_distance)
            elif cls == 'people':
                object_distance = human_height_coefficient / heights[i]
                distance_estimations.append(object_distance)
            elif cls == 'human carrier':
                object_distance = human_carrier_height_coefficient / heights[i]
                distance_estimations.append(object_distance)
                # print(object_distance, heights[i])
            elif cls == 'forklift':
                object_distance = forklift_height_coefficient / heights[i]
                distance_estimations.append(object_distance)
            #far_check[i] = object_distance < max_distances[cls]

        for i, box_i in enumerate(boxes):
            # check whether it is in a forklift
            for j, box_j in enumerate(boxes):
                if classes[j] == 'forklift' and classes[i] == 'people':
                    p1 = (box_i[0] - 20, box_i[1] - 20)
                    p2 = (box_i[0] - 20, box_i[1]+box_i[3] + 20)
                    p3 = (box_i[0] + box_i[2] + 20, box_i[1] - 20)
                    p4 = (box_i[0] + box_i[2] + 20, box_i[1]+box_i[3] + 20)
                    inside_bbox = is_inside(p1, box_j) and is_inside(p2, box_j) and is_inside(p3, box_j) and is_inside(p4, box_j)
                    if inside_bbox:
                        distance_estimations[i] = distance_estimations[j]
            if distance_estimations[i] > max_distances[classes[i]]:
                classes[i] = 'far object'

    suspended_threshold = suspended_threshold_wharf if wharf else suspended_threshold_hatch
    for i in range(len(boxes)):
        if classes[i] in ['suspended lean object', 'people', 'chain'] + ignored_classes + ['far object']:
            continue
        x, y, w, h = boxes[i]
        if y <= suspended_threshold:
            classes[i] = 'suspended lean object'
    
    # only for converting chain to suspended lean object. if the chain is ignored, no need for it!
    if not 'suspended lean object' in classes and not 'chain' in ignored_classes:
        for i, cls in enumerate(classes):
            if cls == 'chain':
                x, y, w, h = boxes[i]
                bottom1 = (x, y+h)
                bottom2 = (x+w, y+h)
                if y <= suspended_threshold:
                    not_inside = True
                    for j, box in enumerate(boxes):
                        if classes[j] != 'people' and i != j:
                            if is_inside(bottom1, box) or is_inside(bottom2, box):
                                print(bottom1, bottom2, box)
                                not_inside = False
                    if not_inside:
                        classes[i] = 'suspended lean object'

    for i, box in enumerate(boxes):
        class_name = classes[i]
        track_id = ids[i]
        cur_distance = int(distance_estimations[i]) if distance_check else -1
        if (not class_name in ignored_classes) and class_name !='far object':
            xyxy = xywh2xyxy(box)
            original_label = f'{class_name} #{track_id}'
            label = f'{original_label} {cur_distance} CM' if distance_check else original_label
            #plot_one_box(xyxy, im0, label=label,color=get_color_for(original_label), line_thickness=3)
    return boxes, classes, distance_estimations,ids
"""
def update_tracks(tracker, im0, width, height, ignored_classes, suspended_threshold_hatch, suspended_threshold_wharf, suspended_threshold_wharf_side, angle, distance_check, wharf, no_action, no_nested):
    if no_action:
        return [], [], [],[]
    max_distances = {
        'forklift': constants.MAX_DISTANCE_FOR_FORKLIFT,
        'suspended lean object': constants.MAX_DISTANCE_FOR_SUSPENDED_LEAN_OBJECT,
        'chain': constants.MAX_DISTANCE_FOR_CHAIN,
        'people': constants.MAX_DISTANCE_FOR_PEOPLE,
        'human carrier': constants.MAX_DISTANCE_FOR_HUMAN_CARRIER,
    }

    human_height_coefficient = constants.HUMAN_HEIGHT_COEFFICIENT
    forklift_height_coefficient = constants.FORKLIFT_HEIGHT_COEFFICIENT
    human_carrier_height_coefficient = constants.HUMAN_CARRIER_HEIGHT_COEFFICIENT

    boxes = []
    classes = []
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
        detections.append(class_name)
        ids.append(track.track_id)

    suspended_threshold = suspended_threshold_wharf if wharf else suspended_threshold_hatch
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        if classes[i] in ['suspended lean object', 'people', 'chain'] + ignored_classes:
            continue
        side_limit_check = True
        if wharf:
            if angle == 'left':
                side_limit_check = (x + w <= suspended_threshold_wharf_side)
            elif angle == 'right':
                side_limit_check = (x >= suspended_threshold_wharf_side)
        if y + h <= suspended_threshold and side_limit_check:
            if h / height >= 0.08:
                classes[i] = 'suspended lean object'
            """for j, chain_candidate in enumerate(boxes):
                if not classes[j] == 'chain':
                    continue
                x_chain, y_chain, w_chain, h_chain = chain_candidate
                if x_chain >= x and x_chain <= x+w: # center of the chain is aligned with the bbox
                    if (y - (y_chain + h_chain)) / height <= 0.1: # y-axis check for bottom of chain and top of object bbox
                        classes[i] = 'suspended lean object'"""
    
    for i in range(len(boxes)):
        class_name = classes[i]
        xywh = boxes[i]
        if class_name in ignored_classes:
            continue
        if class_name in ['people', 'human carrier', 'forklift']:
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
    if not 'suspended lean object' in classes and not 'chain' in ignored_classes:
        for i, cls in enumerate(classes):
            if cls == 'chain':
                x, y, w, h = boxes[i]
                bottom1 = (x, y+h)
                bottom2 = (x+w, y+h)
                if y <= suspended_threshold:
                    not_inside = True
                    for j, box in enumerate(boxes):
                        if classes[j] != 'people' and i != j:
                            if is_inside(bottom1, box) or is_inside(bottom2, box):
                                # print(bottom1, bottom2, box)
                                not_inside = False
                    if not_inside:
                        classes[i] = 'suspended lean object'

    try:
        if no_nested:
            nested_ids = []
            for i, box_i in enumerate(boxes):
                # check whether it is in a forklift or a human carrier
                for j, box_j in enumerate(boxes):
                    if detections[j] == 'forklift' and detections[i] == 'forklift' and i != j:
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
    if not 'people' in classes or not wharf:
        distance_check = False

    distance_estimations = []
    if distance_check:
        max_people_height = -1.0
        i = 0
        while i < len(classes):
            if classes[i] == 'people' and heights[i] > max_people_height:
                max_people_height = heights[i]
            i+=1
        first_object_distance = human_height_coefficient / max_people_height
        first_objects = {cls: -1.0 for cls in classes if not cls in ['people', 'human carrier', 'forklift']}
        for i, cls in enumerate(classes):
            if not cls in ['people', 'human carrier', 'forklift']:
                first_objects[cls] = max(first_objects[cls], areas[i])
        for i, cls in enumerate(classes):
            if not cls in ['people', 'human carrier', 'forklift']:
                reference_area = first_objects[cls]
                reference_distance = first_object_distance
                object_area = areas[i]
                object_distance = np.sqrt(np.square(reference_distance) * reference_area / object_area)
                distance_estimations.append(object_distance)
            elif cls == 'people':
                object_distance = human_height_coefficient / heights[i]
                distance_estimations.append(object_distance)
            elif cls == 'human carrier':
                object_distance = human_carrier_height_coefficient / heights[i]
                distance_estimations.append(object_distance)
                # print(object_distance, heights[i])
            elif cls == 'forklift':
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
                    if classes[j] in ['forklift', 'human carrier'] and classes[i] == 'people':
                        p1 = (box_i[0] , box_i[1] )
                        p2 = (box_i[0] , box_i[1]+box_i[3] )
                        p3 = (box_i[0] + box_i[2] , box_i[1] )
                        p4 = (box_i[0] + box_i[2] , box_i[1]+box_i[3] )
                        inside_bbox = is_inside(p1, box_j) and is_inside(p2, box_j) and is_inside(p3, box_j) and is_inside(p4, box_j)
                        if inside_bbox:
                            distance_estimations[i] = distance_estimations[j]
                if distance_estimations[i] > max_distances[classes[i]] or (classes[i] != 'people' and (box_i[3] * box_i[2]) / (height * width) < 0.0025):
                    classes[i] = 'far object'
            except:
                continue

    for i, box in enumerate(boxes):
        class_name = classes[i]
        detection = detections[i]
        track_id = ids[i]
        cur_distance = int(distance_estimations[i]) if distance_check else -1
        if (not class_name in ignored_classes) and class_name !='far object' and class_name != 'nested object':
            xyxy = xywh2xyxy(box)
            original_label = f'{detection} #{track_id}'
            label = f'{original_label} {cur_distance} CM' if distance_check else original_label
            #plot_one_box(xyxy, im0, label=label,color=get_color_for(original_label), line_thickness=3)
    return boxes, classes, distance_estimations,ids

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
    print('Detection starts...')
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

def detect(opt):
    global inf_1, inf_2, inf_3, inf_4
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
    yolov5_engine = Yolov5_IV() if constants.USE_IV_MODEL else Yolov5()
    global names
    names = yolov5_engine.get_names()

    # initialize tracker
    tracker = Tracker(metric)

    source, imgsz = opt.source, opt.img_size

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                    exist_ok=opt.exist_ok))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device('')

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz)
    all_detections, height, width, fps = get_all_detections_yolov5(source, yolov5_engine, opt.budget)
    frame_count = 0
    distance_tracker = None
    calibrated_frames = 0
    print('Safety zone detection starts...')
    for p, im0, vid_cap in dataset:
        if type(im0) == type(None):
            continue
        if frame_count == len(all_detections):
            break
        c1 = time.time()
        if frame_count == 0:
            distance_tracker = DistanceTracker(im0, opt.source, height, width, fps, ignored_classes, opt.danger_zone_width_threshold, opt.danger_zone_height_threshold, opt.wharf, opt.angle, save_dir)
            suspended_threshold_hatch, suspended_threshold_wharf, suspended_threshold_wharf_side = distance_tracker.get_suspended_threshold()
        if calibrated_frames < 10:
            if calibrated_frames == 0:
                distance_tracker.clear_calibration_count()
            calib_bboxes = []
            for bbox in all_detections[frame_count]:
                if bbox[-1] == names.index('people'): 
                    calib_bboxes.append(bbox.detach().cpu())
            if len(calib_bboxes) > 0:
                calibrated_frames += 1
                distance_tracker.calibrate_lengths(calib_bboxes)
        else:
            calibrated_frames = (calibrated_frames + 1) % 30

        pred = all_detections[frame_count].unsqueeze(0)
        c2 = time.time()
        inf_1 += (c2-c1)
        # Process detections
        for i, det in enumerate(pred):  # detections per image

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

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
                features = encoder(im0, bboxes)
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
                inf_2 += (c3-c2)
                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                # update tracks
                bboxes, classes, distance_estimations,ids = update_tracks(tracker, im0, width, height, ignored_classes, suspended_threshold_hatch, suspended_threshold_wharf, suspended_threshold_wharf_side, opt.angle, opt.distance_check, opt.wharf, not hasattr(distance_tracker, 'distance_w'), opt.no_nested)
                #print("length of bboxes {}".format(bboxes))
                #print("length of ids {}".format(len(ids)))
                c4 = time.time()
                inf_3 += (c4-c3)
            # update distance tracker
            distance_tracker.calculate_distance(bboxes, classes, distance_estimations, im0, frame_count,ids)
            # Print time (inference + NMS)
            c5 = time.time()
            inf_4 += (c5-c4)

            # Save results (image with detections)
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(im0)

            frame_count = frame_count+1

    print(f"Results saved to {save_dir}")
    elapsed_time = time.time() - t0
    print(f'Done. ({elapsed_time:.3f}s, fps: {frame_count/elapsed_time:.3f})', inf_1, inf_2, inf_3, inf_4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=float, default=-1.0)
    parser.add_argument('--source', required=True)
    parser.add_argument('--wharf', action='store_true')
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
    parser.add_argument('--max_cosine_distance', type=float, default=0.4,
                        help='Gating threshold for cosine distance metric (object appearance).')
    parser.add_argument('--nn_budget', type=int, default=None,
                        help='Maximum size of the appearance descriptors allery. If None, no budget is enforced.')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt)
