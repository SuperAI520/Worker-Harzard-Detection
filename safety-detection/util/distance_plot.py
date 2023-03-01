# imports
import cv2
import numpy as np
import constants
import os.path as osp
import time
import requests
import math

# Function to draw Bird Eye View for region of interest(ROI). Red, Yellow, Green points represents risk to human. 
# Red: High Risk
# Yellow: Low Risk
# Green: No Risk

text_scale = 1.5
text_thickness = 2
line_thickness = 3
db_push=True
db_freq_frames=20
viol_thresh_fl_fh=0.9


    
# Function to draw bounding boxes according to risk factor for humans in a frame and draw lines between
# boxes according to risk factor between two humans.
# Red: High Risk
# Yellow: Low Risk
# Green: No Risk 
def social_distancing_view(frame, cargo_ids, pairs, boxes, inversed_pts, heights,ids,all_violations,frame_id,fps,output_dir,wharf):
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    yellow = (0, 255, 255)
    vid_save_path=output_dir
    snap_path=output_dir
    risk_count = 0
    thr_frames = fps * 1  # Threshold of frames at which the violation persists : 1s
    if wharf:
        vessel_area="wharf"
    else:
        vessel_area="hatch"

    for cargo_id in cargo_ids:
        xi, yi, wi, hi = boxes[cargo_id]
        frame = cv2.rectangle(frame,(int(xi),int(yi)),(int(xi+wi),int(yi+hi)),blue,3)
        
    new_sload_prox = False
    tl = 3
    tf = max(tl - 1, 1) 
    for pair in pairs:
        i, j, dist, danger = pair
        xi, yi, wi, hi = boxes[i]
        proj_center = (int(inversed_pts[i][0]), int(inversed_pts[i][1]))
        bbox_center = (int(xi+wi/2), int(yi+hi/2))
        xj, yj, wj, hj = boxes[j]
        pt = inversed_pts[j]
        obj_id=ids[j]
        if obj_id not in all_violations.keys():
            violation_dict = {'first_frame_id':frame_id,'sload_last_pushed_frame_id':0,'sload_prox':False,'fall_fh':False,'fall_fh__last_pushed_frame_id':0, 'sload_prox_start_frame_id': 0, 'sload_prox_frame_buffers': {'last_frame_id': 0, 'count': 0}, 'fall_fh_start_frame_id':0, 'fall_fh_frame_buffers':{'last_frame_id': 0, 'count': 0}}
            all_violations[obj_id] = violation_dict
        #frame = cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, yellow, 10)
        if all_violations[obj_id]['sload_prox_frame_buffers']['last_frame_id'] + 1 != frame_id:
            all_violations[obj_id]['sload_prox_frame_buffers']['count'] = 0

        if danger:
            all_violations[obj_id]['sload_prox_frame_buffers']['count'] += 1
            all_violations[obj_id]['sload_prox_frame_buffers']['last_frame_id'] = frame_id
        else:
            all_violations[obj_id]['sload_prox_frame_buffers']['count'] = 0

        if all_violations[obj_id]['sload_prox_frame_buffers']['count'] >= thr_frames:
            frame = cv2.rectangle(frame,(int(xj),int(yj)),(int(xj+wj),int(yj+hj)),red,3,lineType=cv2.LINE_AA)
            label = 'Suspended_Load'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=tf)[0]
            frame = cv2.rectangle(frame, (int(xj),int(yj)),(int(xj+t_size[0]),int(yj) - t_size[1] - 3), red, -1, cv2.LINE_AA)
            frame = cv2.putText(frame, label, (int(xj), int(yj) - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            if all_violations[obj_id]['sload_prox_start_frame_id'] == 0:
                new_sload_prox = True
                all_violations[obj_id]['sload_prox_start_frame_id'] = frame_id
            else:
                time_period = (frame_id - all_violations[obj_id]['sload_prox_start_frame_id']) / fps
                if time_period > 60: # ignore new violation within 60s
                    new_sload_prox = True
                    all_violations[obj_id]['sload_prox_start_frame_id'] = frame_id

            all_violations[obj_id]['sload_prox'] = True
        else:
            all_violations[obj_id]['sload_prox'] = False

    new_size = (int(frame.shape[1]*constants.OUTPUT_RES_RATIO), int(frame.shape[0]*constants.OUTPUT_RES_RATIO))
    frame = cv2.resize(frame, new_size)
    
    # pad = np.full((140,frame.shape[1],3), [110, 110, 100], dtype=np.uint8)
    # cv2.putText(pad, "Red bounding boxes denote people in danger zone.", (50, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 0), 2)
    # cv2.putText(pad, "-- RISK : " + str(risk_count) + " people", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    # frame = np.vstack((frame,pad))
            
    return frame, new_sload_prox, all_violations

def draw_danger_zones(img, reversed_danger_zones):
    for danger_zone in reversed_danger_zones:
        pts = np.array(danger_zone, np.int32)
        img = cv2.polylines(img, [pts], True, (0, 0, 255), thickness=10)

    return img

def point_on_line(a, b, p):
    ap = p - a
    ab = b - a
    result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return result

def get_min_distance(p0, p1, p2, shape):
    projected = point_on_line(p1, p2, p0)
    x1, x2, x0 = p1[0], p2[0], projected[0]
    y1, y2, y0 = p1[1], p2[1], projected[1]

    if (x1==x2 and (x1 == shape[1] - 1 or x1 == 0)) or (y1==y2 and (y1 == shape[0] - 1 or y1 == 0)) :
        distance = shape[0]
        return distance

    # slope = (y2 - y1) / (x2 - x1)
    # projected_on = (y0 - y1) == slope * (x0 - x1)
    projected_between = (min(x1, x2) <= x0 <= max(x1, x2)) and (min(y1, y2) <= y0 <= max(y1, y2))
    # on_and_between = (projected_on and projected_between)

    if projected_between:
        distance=np.linalg.norm(np.cross(p2-p1,p0-p1)/np.linalg.norm(p2-p1))/100
    else:
        eDistance1 = np.linalg.norm(p0-p1) / 100
        eDistance2 = np.linalg.norm(p0-p2) / 100            
        distance = min(eDistance1, eDistance2)

    return distance
    
def calculate_edge_to_person(roi_edge,frame, ori_shape, boxes,classes,frame_id, thr_f_h, all_violations,ids,output_dir,fps):
    #roi_pts = np.array(self.reference_points, np.int32)\
    green = (0, 255, 0)
    text_scale = 1.5
    text_thickness = 2
    line_thickness = 3
    red = (0, 0, 255)
    vid_save_path=output_dir
    snap_path=output_dir
    risk_count = 0
    vessel_area="hatch"
    viol_thresh_fl_fh = thr_f_h
    new_Fall_F_H = False
    thr_frames = fps * 1 # Threshold of frames at which the violation persists : 1s
    #roi_edge = [self.reference_points[0],self.reference_points[1]]
    tl = 3
    tf = max(tl - 1, 1) 
    for i in range(len(boxes)):
        if classes[i]=='People':
            #i, j, dist, danger = pair
            #xi, yi, wi, hi = boxes[i]
            xj, yj, wj, hj = boxes[i]
            p0=np.array([xj + wj / 2, yj + hj])
            # p1=np.array(list(roi_edge[0]))
            # p2=np.array(list(roi_edge[1]))

            distance = ori_shape[0]
            for edge_pts in roi_edge:
                inside = cv2.pointPolygonTest(edge_pts, p0, False)
                if inside < 0:
                    continue
                
                for k in range(len(edge_pts)):
                    if k + 1 == len(edge_pts):
                        distance = get_min_distance(p0, edge_pts[k][0], edge_pts[0][0], ori_shape)
                    else:
                        distance = get_min_distance(p0, edge_pts[k][0], edge_pts[k+1][0], ori_shape)

                    if round(float(distance),2)<viol_thresh_fl_fh:
                        break

                if round(float(distance),2)<viol_thresh_fl_fh:
                        break
            # distance=np.linalg.norm(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))/100

            obj_id=ids[i]
            if obj_id not in all_violations.keys():
                violation_dict = {'first_frame_id':frame_id,'sload_last_pushed_frame_id':0,'sload_prox':False,'fall_fh':False,'fall_fh__last_pushed_frame_id':0, 'sload_prox_start_frame_id': 0, 'sload_prox_frame_buffers': {'last_frame_id': 0, 'count': 0}, 'fall_fh_start_frame_id':0, 'fall_fh_frame_buffers':{'last_frame_id': 0, 'count': 0}}
                all_violations[obj_id] = violation_dict

            if all_violations[obj_id]['fall_fh_frame_buffers']['last_frame_id'] + 1 != frame_id:
                all_violations[obj_id]['fall_fh_frame_buffers']['count'] = 0

            if round(float(distance),2)<viol_thresh_fl_fh:
                all_violations[obj_id]['fall_fh_frame_buffers']['count'] += 1
                all_violations[obj_id]['fall_fh_frame_buffers']['last_frame_id'] = frame_id
            else:
                all_violations[obj_id]['fall_fh_frame_buffers']['count'] = 0

            if all_violations[obj_id]['fall_fh_frame_buffers']['count'] >= thr_frames:
                frame = cv2.rectangle(frame,(int(xj),int(yj)),(int(xj+wj),int(yj+hj)),red,3,lineType=cv2.LINE_AA)
                label = 'Worker_on_Edge'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=tf)[0]
                frame=cv2.rectangle(frame, (int(xj),int(yj)),(int(xj+t_size[0]),int(yj) - t_size[1] - 3), red, -1, cv2.LINE_AA)
                frame = cv2.putText(frame, label, (int(xj), int(yj) - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                
                if all_violations[obj_id]['fall_fh_start_frame_id'] == 0:
                    new_Fall_F_H = True
                    all_violations[obj_id]['fall_fh_start_frame_id'] = frame_id
                else:
                    time_period = (frame_id - all_violations[obj_id]['fall_fh_start_frame_id']) / fps
                    if time_period > 60: # ignore new violation within 60s
                        new_Fall_F_H = True
                        all_violations[obj_id]['fall_fh_start_frame_id'] = frame_id
                
                all_violations[obj_id]['fall_fh']= True
            else:
                all_violations[obj_id]['fall_fh']= False
    new_size = (int(frame.shape[1]*constants.OUTPUT_RES_RATIO), int(frame.shape[0]*constants.OUTPUT_RES_RATIO))
    frame = cv2.resize(frame, new_size)
    return frame, new_Fall_F_H, all_violations
def no_action(frame):
    new_size = (int(frame.shape[1]*constants.OUTPUT_RES_RATIO), int(frame.shape[0]*constants.OUTPUT_RES_RATIO))
    frame = cv2.resize(frame, new_size)
    
    # pad = np.full((140,frame.shape[1],3), [110, 110, 100], dtype=np.uint8)
    # cv2.putText(pad, "No action is taken since no people is detected!", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 0), 3)
    # frame = np.vstack((frame,pad))

    return frame