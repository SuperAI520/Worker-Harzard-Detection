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

#vessel_area="wharf"
folder_timestamp=str(int(time.time()))
text_scale = 1.5
text_thickness = 2
line_thickness = 3
db_push=True
db_freq_frames=20
viol_thresh_fl_fh=0.9
def bird_eye_view(frame, pairs, bottom_points, scale_w, scale_h):
    h = frame.shape[0]
    w = frame.shape[1]

    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    white = (200, 200, 200)

    blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
    blank_image[:] = white
    warped_pts = []
    for pair in pairs:
        i, j, dist, danger = pair
        blank_image = cv2.circle(blank_image, (int(bottom_points[i][0]  * scale_w), int(bottom_points[i][1] * scale_h)), 5, blue, 10)
        if danger:
            blank_image = cv2.circle(blank_image, (int(bottom_points[j][0]  * scale_w), int(bottom_points[j][1] * scale_h)), 5, red, 10)
        else:
            blank_image = cv2.circle(blank_image, (int(bottom_points[j][0]  * scale_w), int(bottom_points[j][1] * scale_h)), 5, green, 10)   
        
    return blank_image
    
# Function to draw bounding boxes according to risk factor for humans in a frame and draw lines between
# boxes according to risk factor between two humans.
# Red: High Risk
# Yellow: Low Risk
# Green: No Risk 
def social_distancing_view(frame, pairs, boxes, inversed_pts, heights,ids,all_violations,frame_id,fps,output_dir,wharf):
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    yellow = (0, 255, 255)
    vid_save_path=output_dir
    snap_path=output_dir
    risk_count = 0
    if wharf:
        vessel_area="wharf"
    else:
        vessel_area="hatch"
    for pair in pairs:
        i, j, dist, danger = pair
        xi, yi, wi, hi = boxes[i]
        proj_center = (int(inversed_pts[i][0]), int(inversed_pts[i][1]))
        bbox_center = (int(xi+wi/2), int(yi+hi/2))
        frame = cv2.rectangle(frame,(int(xi),int(yi)),(int(xi+wi),int(yi+hi)),blue,2)
        #frame = cv2.circle(frame, bbox_center, 5, yellow, 10)
        #frame = cv2.circle(frame, proj_center, 5, yellow, 10)
        #frame = cv2.line(frame, proj_center, bbox_center, red, 2)
        #text_place = (int(bbox_center[0]*0.5 + proj_center[0]*0.5), int(bbox_center[1]*0.5 + proj_center[1]*0.5))
        #frame = cv2.putText(frame, f'{heights[i]} CM', text_place, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
        xj, yj, wj, hj = boxes[j]
        pt = inversed_pts[j]
        obj_id=ids[j]
        if obj_id not in all_violations.keys():
            violation_dict = {'first_frame_id':frame_id,'sload_last_pushed_frame_id':0,'sload_prox':False,'fall_fh':False,'fall_fh__last_pushed_frame_id':0}
            all_violations[obj_id] = violation_dict
        #frame = cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, yellow, 10)
        if danger:
            frame = cv2.rectangle(frame,(int(xj),int(yj)),(int(xj+wj),int(yj+hj)),red,2)
            frame=cv2.putText(frame, 'sload_prox', (int(xj), int(yj)-10), cv2.FONT_HERSHEY_PLAIN, text_scale,red,thickness=text_thickness) 
            #frame = cv2.line(frame, proj_center, (int(xj+wj/2), int(yj+hj/2)), red, 2)
            #frame = cv2.line(frame, proj_center, (int(xj+wj/2), int(yj+hj)), red, 2)
            risk_count += 1
            #violation_dict = {'first_frame_id':frame_id,'sload_last_pushed_frame_id':0,'sload_prox':True,'fall_fh':False,'fall_fh__last_pushed_frame_id':0}
            all_violations[obj_id]['sload_prox'] = True
            #all_violations[obj_id]['fall_fh__last_pushed_frame_id']=frame_id
        else:
            #frame = cv2.rectangle(frame,(int(xj),int(yj)),(int(xj+wj),int(yj+hj)),green,2)
            #violation_dict = {'first_frame_id':frame_id,'sload_last_pushed_frame_id':0,'sload_prox':False,'fall_fh':False,'fall_fh__last_pushed_frame_id':0}
            #all_violations[obj_id] = violation_dict
            all_violations[obj_id]['sload_prox'] = False
        """for pt in inversed_pts:
        frame = cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, yellow, 10)"""
        if obj_id in all_violations.keys():
            viol_text = ''
            if all_violations[obj_id]['sload_prox'] == True :
                viol_text += 'sload_prox,'
                viol_flag = True
                if (db_push==True) and ((frame_id - all_violations[obj_id]["sload_last_pushed_frame_id"]) / fps >= db_freq_frames):
                    s_img_name = osp.splitext(osp.basename(vid_save_path))[0] + "_" + str(int(time.time())) + str (np.random.randint(100)) + ".png"
                    snap_imgname = osp.join(snap_path, s_img_name) 
                    cv2.putText(frame, 'sload_prox', (int(xj), int(yj)-10), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                                        thickness=text_thickness) 
                    cv2.rectangle(frame,(int(xj),int(yj)),(int(xj+wj),int(yj+hj)),red,thickness=line_thickness)
                    cv2.imwrite(snap_imgname, frame)
                    if push_alert(frame_id=frame_id,fps=fps,vessel_area=vessel_area,viol_cat='sload_prox',
                        viol='sload_prox',folder_timestamp=folder_timestamp,vid_save_path=vid_save_path,s_img_name=s_img_name):
                        
                        all_violations[obj_id]["sload_last_pushed_frame_id"] = frame_id
    new_size = (int(frame.shape[1]*constants.OUTPUT_RES_RATIO), int(frame.shape[0]*constants.OUTPUT_RES_RATIO))
    frame = cv2.resize(frame, new_size)
    
    pad = np.full((140,frame.shape[1],3), [110, 110, 100], dtype=np.uint8)
    cv2.putText(pad, "Red bounding boxes denote people in danger zone.", (50, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 0), 2)
    cv2.putText(pad, "-- RISK : " + str(risk_count) + " people", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    frame = np.vstack((frame,pad))
            
    return frame,all_violations

def draw_danger_zones(img, reversed_danger_zones):
    for danger_zone in reversed_danger_zones:
        pts = np.array(danger_zone, np.int32)
        img = cv2.polylines(img, [pts], True, (0, 0, 255), thickness=10)

    return img
def push_alert(frame_id,fps,vessel_area,viol_cat,viol,folder_timestamp,vid_save_path,s_img_name) -> bool:
    return False
    db_push_api = "http://jp.groundup.ai:5566/db/push_violations/v1/"
    # secrets for api access
    APIKEY = "NDI2NzRmMDM2Y2U1ZGZiNTg1M2YxMDk0"
    APINAME = "DatabaseApi"

    db_data = { "timestamp": time.time(),
                "offset_seconds": (frame_id / fps),
                "vessel": "VISION_211",
                "berth": "U2K14",
                "area": vessel_area,
                "camera": "Camera 15",
                "violation_category": viol_cat,
                "violation_sub_category": viol,
                "video_link": "/home/webapp_suvrat/cv_videos/kpi_vis/" + folder_timestamp + "/" + osp.basename(vid_save_path),
                "image_link": "/home/webapp_suvrat/snapshots/" + s_img_name 
            }
    #print(db_data)
    response = requests.post(db_push_api, json=db_data,params={APINAME:APIKEY})
    # print(response.json())
    try:
        response = response.json()
    except ValueError:
        return False

    try:
        if len(response['violation_id']) > 1:
            print(response)
            return True
        else:
            return False
    except:
        print("Issue at pushing data to db")
        return False

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
    
def calculate_edge_to_person(roi_edge,frame, ori_shape, boxes,classes,frame_id,all_violations,ids,output_dir,fps):
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
    #roi_edge = [self.reference_points[0],self.reference_points[1]]
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
                violation_dict = {'first_frame_id':frame_id,'sload_last_pushed_frame_id':0,'sload_prox':False,'fall_fh':False,'fall_fh__last_pushed_frame_id':0}
                all_violations[obj_id] = violation_dict
            if round(float(distance),2)<viol_thresh_fl_fh:
                #color = get_color(abs(obj_id))
                frame=cv2.rectangle(frame,(int(xj),int(yj)),(int(xj+wj),int(yj+hj)), color=(0,0,255 ), thickness=line_thickness)
                #frame=cv2.putText(frame, "Fall_F_H {}".format(round(float(distance),2)), (int(xj), int(yj)-10), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),thickness=text_thickness)
                frame=cv2.putText(frame, "Fall_F_H", (int(xj), int(yj)-10), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),thickness=text_thickness)
                #violation_dict = {'first_frame_id':frame_id,'sload_last_pushed_frame_id':0,'sload_prox':True,'fall_fh':False,'fall_fh__last_pushed_frame_id':0}
                all_violations[obj_id]['fall_fh']= True
            else:
                #frame=cv2.rectangle(frame,(int(xj),int(yj)),(int(xj+wj),int(yj+hj)), color=green, thickness=line_thickness)
                #frame=cv2.putText(frame, "Fall_F_H {}".format(round(float(distance),2)), (int(xj), int(yj)-10), cv2.FONT_HERSHEY_PLAIN, text_scale, green,thickness=text_thickness)
                all_violations[obj_id]['fall_fh']= False
            if obj_id in all_violations.keys():
                viol_text = ''
                if all_violations[obj_id]['fall_fh'] == True :
                    #print("Fall from height")
                    viol_text += 'Fall_F_H,'
                    viol_flag = True
                    if (db_push==True) and ((frame_id - all_violations[obj_id]["fall_fh__last_pushed_frame_id"]) / fps >= db_freq_frames):
                        s_img_name = osp.splitext(osp.basename(vid_save_path))[0] + "_" + str(int(time.time())) + str (np.random.randint(100)) + ".png"
                        snap_imgname = osp.join(snap_path, s_img_name) 
                        #frame=cv2.putText(frame, 'Fall_F_H', (int(xj), int(yj)-10), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                                        #thickness=text_thickness) 
                        #frame=cv2.rectangle(frame,(int(xj),int(yj)),(int(xj+wj),int(yj+hj)),red,thickness=line_thickness)
                        cv2.imwrite(snap_imgname, frame)
                        if push_alert(frame_id=frame_id,fps=fps,vessel_area=vessel_area,viol_cat='Fall_F_H',
                            viol='Fall_F_H',folder_timestamp=folder_timestamp,vid_save_path=vid_save_path,s_img_name=s_img_name):
                            all_violations[obj_id]["fall_fh__last_pushed_frame_id"] = frame_id
                            print("Fall from height pushed to db")
    new_size = (int(frame.shape[1]*constants.OUTPUT_RES_RATIO), int(frame.shape[0]*constants.OUTPUT_RES_RATIO))
    frame = cv2.resize(frame, new_size)
    return frame,all_violations
def no_action(frame):
    new_size = (int(frame.shape[1]*constants.OUTPUT_RES_RATIO), int(frame.shape[0]*constants.OUTPUT_RES_RATIO))
    frame = cv2.resize(frame, new_size)
    
    pad = np.full((140,frame.shape[1],3), [110, 110, 100], dtype=np.uint8)
    cv2.putText(pad, "No action is taken since no people is detected!", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 0), 3)
    frame = np.vstack((frame,pad))

    return frame