# imports
import cv2
import numpy as np
import time
import argparse
import util.distance_utils as utills
import util.distance_plot as plot
import constants
import torch
import torchvision.ops.boxes as bops
import threading
import util.threads as threads
from util.db_manager import s3_sqs_handler, handle_annot_frames_buffer

class DistanceTracker:
    def __init__(self, frame, source, height, width, fps, ignored_classes, danger_zone_width_threshold, danger_zone_height_threshold, work_area, height_edges, wharf, output_dir, save_result):
        self.output_dir = output_dir
        # Get video height, width and fps
        self.height = height
        self.width = width
        self.save_result = save_result
        self.filename=source.split('/')[-1].split('.')[0]
        self.edge_points = height_edges
        self.reference_points = work_area
        self.wharf = wharf
        self.calibrate_reference_area(source.split('/')[-1])
        if self.save_result:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.output_movie = cv2.VideoWriter(f"{output_dir}/{source.split('/')[-1].split('.')[0]}_dist.avi", fourcc, fps, (int(self.width*constants.OUTPUT_RES_RATIO), int(self.height*constants.OUTPUT_RES_RATIO)))
        self.calibrated_frames = 0

        self.wharf_calibrated_frames = 0
        self.wharf_avg_person_height = 0

        self.hatch_human_height = 0

        self.ignored_classes = ignored_classes
        self.danger_zone_height_threshold = danger_zone_height_threshold
        self.all_violations=dict()
        self.fps=fps

    def update_workarea(self, work_area):
        self.reference_points = work_area

    def update_edgepoints(self, edge_points):
        self.edge_points = edge_points

    def get_suspended_threshold(self):
        if len(self.reference_points) == 0:
            return 0, 0
        return self.suspended_threshold_hatch, self.suspended_threshold_wharf

    def calibrate_reference_area(self, video_file):
        if len(self.reference_points) == 0:
            return

        if self.wharf:
            self.suspended_threshold_wharf, self.suspended_threshold_hatch = self.height / 2, 0
            return

        # print(self.reference_points)
        src = np.float32(np.array(self.reference_points))
        dst = np.float32([[0, self.height], [self.width, self.height], [self.width, 0], [0, 0]])

        self.perspective_transform = cv2.getPerspectiveTransform(src, dst)
        self.inverse_perspective_transform = np.linalg.pinv(self.perspective_transform)
        roi_upper_y = min(self.reference_points[2][1], self.reference_points[3][1])
        roi_long_edge = max(self.reference_points[1][1] - self.reference_points[2][1], self.reference_points[0][1] - self.reference_points[3][1])
        candidate_y = roi_long_edge
        self.suspended_threshold_hatch = max(candidate_y, roi_upper_y)
        self.suspended_threshold_wharf = self.height / 2
        
    def calibrate_person_height_wharf(self, frame, bboxes, landing_Y, ground_height):
        pts = []
        n_people = len(bboxes)
        valid_range = ground_height // 10
        for bbox in bboxes:
            top = [(bbox[0]+bbox[2])/2, bbox[1]]
            bottom = [(bbox[0]+bbox[2])/2, bbox[3]]
            
            pts += [top, bottom]
            
        distance_h = 0
        valid_num = 0
        min_human_height = self.height
        if self.wharf_avg_person_height == 0:
            for bbox in bboxes:
                if abs(landing_Y - bbox[3]) > valid_range:
                    continue
                distance_h += np.abs(bbox[3] - bbox[1])
                valid_num += 1
            if valid_num > 0:
                self.wharf_avg_person_height = distance_h / valid_num
                self.wharf_calibrated_frames += 1
        else:
            for bbox in bboxes:
                if abs(landing_Y - bbox[3]) > valid_range:
                    continue
                person_height = np.abs(bbox[3] - bbox[1])
                # frame = cv2.line(frame, (int((bbox[0]+bbox[2])/2), int(bbox[3])), (int((bbox[0]+bbox[2])/2), int(bbox[1])), (255, 255, 255), 5)
                # frame = cv2.line(frame, (int((bbox[0]+bbox[2])/2 + 7), int(bbox[3] - self.wharf_avg_person_height * 0.7)), (int((bbox[0]+bbox[2])/2) + 7, int(bbox[3])), (0, 0, 255), 5)
                
                if person_height < self.wharf_avg_person_height * 0.7:
                    continue

                distance_h += person_height
                valid_num += 1
            if valid_num > 0:
                distance_h /= valid_num
                wharf_avg_person_height = self.wharf_avg_person_height * self.wharf_calibrated_frames + distance_h
                self.wharf_calibrated_frames += 1
                self.wharf_avg_person_height = wharf_avg_person_height / self.wharf_calibrated_frames

        # frame = cv2.rectangle(frame,(0,landing_Y - valid_range),(self.width - 1, landing_Y + valid_range),(255,0,0),2)
        return frame
        
    def calibrate_lengths(self, bboxes):
        pts = []
        n_people = len(bboxes)
        for bbox in bboxes:
            top = [(bbox[0]+bbox[2])/2, bbox[1]]
            bottom = [(bbox[0]+bbox[2])/2, bbox[3]]
            pts += [top, bottom]
        pts = np.float32(np.array([pts]))
        warped_pt = cv2.perspectiveTransform(pts, self.perspective_transform)[0]
        distance_h = 0
        hatch_human_height = 0
        wharf_human_height = 0
        for i in range(n_people):
            top = warped_pt[2*i]
            bottom = warped_pt[2*i+1]
            distance_h += np.abs(bottom[1] - top[1])
            hatch_human_height += np.abs(bboxes[i][3] - bboxes[i][1])
            wharf_human_height = max(np.abs(bboxes[i][3] - bboxes[i][1]), wharf_human_height)
        distance_h /= n_people
        hatch_human_height /= n_people
        distance_w = distance_h
        if self.calibrated_frames == 0:
            self.distance_h = distance_h
            self.distance_w = distance_w
            self.hatch_human_height = hatch_human_height
            self.wharf_human_height = wharf_human_height
            self.calibrated_frames += 1
        else:
            temp_h = self.distance_h * self.calibrated_frames + distance_h
            temp_w = self.distance_w * self.calibrated_frames + distance_w
            temp_hatch_human_height = self.hatch_human_height * self.calibrated_frames + hatch_human_height
            temp_wharf_human_height = self.wharf_human_height * self.calibrated_frames + wharf_human_height
            self.calibrated_frames += 1
            self.distance_h = temp_h / self.calibrated_frames
            self.distance_w = temp_w / self.calibrated_frames
            self.hatch_human_height = temp_hatch_human_height / self.calibrated_frames
            self.wharf_human_height = temp_wharf_human_height / self.calibrated_frames
    
    def clear_calibration_count(self):
        self.calibrated_frames = 0
        if hasattr(self, 'distance_w'):
            delattr(self, 'distance_w')
            delattr(self, 'distance_h')
            delattr(self, 'wharf_human_height')    
            delattr(self, 'hatch_human_height')    

    def calculate_distance(self, work_area_index, boxes, classes, old_classes, frame, count,ids, thr_f_h, wharf_landing_Y, hatch_reference, db_manager):
        no_action=0
        roi_pts = np.array(self.reference_points, np.int32)
        cv2.polylines(frame, [roi_pts], True, (70, 70, 70), thickness=10)
        #frame1 = np.copy(frame)
        new_sload_prox, new_Fall_F_H = False, False
        if (self.wharf or hasattr(self, 'distance_w')) and work_area_index != -1:
            if self.wharf:
                if wharf_landing_Y > 0:
                    #img = cv2.line(frame, (0, int(wharf_landing_Y)), (0, int(wharf_landing_Y - self.wharf_avg_person_height)), (255, 0, 0), 20)
                    cargo_ids, pairs, project_pts, danger_zones, heights = utills.get_danger_zones_wharf(boxes, wharf_landing_Y, self.wharf_avg_person_height * 0.7, self.reference_points, classes, old_classes, self.width, self.height, self.wharf_avg_person_height)
                    img = plot.draw_danger_zones(frame, danger_zones)
                    img, new_sload_prox, self.all_violations = plot.social_distancing_view(img, cargo_ids, pairs, boxes, project_pts, heights,ids,self.all_violations,count,self.fps,self.filename,self.wharf) #social_distancing_view(img, pairs, boxes, reversed_pts, heights)
                else:
                    img = frame
            else:
                # print('Distance', len(boxes))
                cargo_ids, pairs, warped_pts, danger_zones, heights = utills.get_distances(boxes, self.reference_points, hatch_reference, self.perspective_transform, self.inverse_perspective_transform, classes, old_classes, self.distance_w, self.distance_h, self.width, self.height, self.danger_zone_height_threshold, self.wharf)
                reversed_pts = utills.get_perspective_transform(warped_pts, self.inverse_perspective_transform)
                reversed_danger_zones = utills.get_reversed_danger_zones(danger_zones, self.inverse_perspective_transform)
                img = plot.draw_danger_zones(frame, reversed_danger_zones)
                img, new_sload_prox, self.all_violations = plot.social_distancing_view(img, cargo_ids, pairs, boxes, reversed_pts, heights,ids,self.all_violations,count,self.fps,self.filename,self.wharf) #social_distancing_view(img, pairs, boxes, reversed_pts, heights)

                roi_edge= self.edge_points
                # print(roi_edge)
                thr_f_h = self.hatch_human_height * 0.2 /100
                img, new_Fall_F_H, self.all_violations=plot.calculate_edge_to_person(roi_edge,img, frame.shape, boxes, classes,count, thr_f_h, self.all_violations,ids,self.filename,self.fps)
                #print(self.all_violations)

            if new_sload_prox or new_Fall_F_H:
                viol_text = ''
                if new_sload_prox:
                    viol_text += 'Suspended_Load'
                if new_Fall_F_H:
                    viol_text += 'Worker_on_Edge'

                snap_path, snap_imgname = db_manager.snap_image(img, viol_text)
                # db_manager.record_start(img)
                # s_img_name = snap_imgname.split('/')[-1].split('.')[0]

                s3_sqs_thread = threading.Thread(target=s3_sqs_handler,args=(snap_path, snap_imgname,snap_imgname,viol_text,count,frame.shape[0],frame.shape[1], self.fps, True,))
                s3_sqs_thread.daemon = True
                s3_sqs_thread.start()
                threads.thread_manager.add_thread(s3_sqs_thread)
        else:
            no_action+=1
            print("no action {} and {}".format(no_action,count))
            img = plot.no_action(frame)
        if count != 0:
            if self.save_result:
                self.output_movie.write(img)
            handle_annot_frames_buffer(frame=img,frame_id=count)
            # db_manager.record_update(img)
        if self.save_result:
            if count == 0:
                cv2.imwrite(f"{self.output_dir}/frame%d.jpg" % count, img)