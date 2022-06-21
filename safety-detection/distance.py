# imports
import cv2
import numpy as np
import time
import argparse
import distance_utils as utills
import distance_plot as plot
from keypoint_detection import get_keypoints
import constants
import torch
import torchvision.ops.boxes as bops

class DistanceTracker:
    def __init__(self, frame, source, height, width, fps, ignored_classes, danger_zone_width_threshold, danger_zone_height_threshold, wharf,angle, output_dir):
        self.output_dir = output_dir
        # Get video height, width and fps
        self.height = height
        self.width = width
        self.angle = angle
        self.filename=source.split('/')[-1].split('.')[0]
        self.edge_points = constants.EDGE_AREA_DICT[source.split('/')[-1]]
        self.calibrate_reference_area(source.split('/')[-1])
        self.scale_w, self.scale_h = utills.get_scale(self.width, self.height)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.output_movie = cv2.VideoWriter(f"{output_dir}/{source.split('/')[-1].split('.')[0]}_dist.avi", fourcc, fps, (int(self.width*constants.OUTPUT_RES_RATIO), int(self.height*constants.OUTPUT_RES_RATIO)+constants.REPORT_PAD))
        self.calibrated_frames = 0

        self.ignored_classes = ignored_classes
        self.danger_zone_width_threshold = danger_zone_width_threshold
        self.danger_zone_height_threshold = danger_zone_height_threshold
        self.wharf = wharf
        self.all_violations=dict()
        self.fps=fps


    def get_suspended_threshold(self):
        return self.suspended_threshold_hatch, self.suspended_threshold_wharf, self.suspended_threshold_wharf_side

    def calibrate_reference_area(self, video_file):
        self.reference_points = constants.REFERENCE_AREA_DICT[video_file] #get_keypoints(frame)
        print(self.reference_points)
        src = np.float32(np.array(self.reference_points))
        dst = np.float32([[0, self.height], [self.width, self.height], [self.width, 0], [0, 0]])

        self.perspective_transform = cv2.getPerspectiveTransform(src, dst)
        self.inverse_perspective_transform = np.linalg.pinv(self.perspective_transform)
        roi_upper_y = min(self.reference_points[2][1], self.reference_points[3][1])
        roi_long_edge = max(self.reference_points[1][1] - self.reference_points[2][1], self.reference_points[0][1] - self.reference_points[3][1])
        candidate_y = roi_long_edge
        self.suspended_threshold_hatch = max(candidate_y, roi_upper_y)
        self.suspended_threshold_wharf = self.height / 2
        self.suspended_threshold_wharf_side = max(self.reference_points[0][0], self.reference_points[3][0]) if self.angle == 'right' else min(self.reference_points[1][0], self.reference_points[2][0])
        
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
        wharf_human_height = 0
        for i in range(n_people):
            top = warped_pt[2*i]
            bottom = warped_pt[2*i+1]
            distance_h += np.abs(bottom[1] - top[1])
            wharf_human_height = max(np.abs(bboxes[i][3] - bboxes[i][1]), wharf_human_height)
        distance_h /= n_people
        distance_w = distance_h
        if self.calibrated_frames == 0:
            self.distance_h = distance_h
            self.distance_w = distance_w
            self.wharf_human_height = wharf_human_height
            self.calibrated_frames += 1
        else:
            temp_h = self.distance_h * self.calibrated_frames + distance_h
            temp_w = self.distance_w * self.calibrated_frames + distance_w
            temp_wharf_human_height = self.wharf_human_height * self.calibrated_frames + wharf_human_height
            self.calibrated_frames += 1
            self.distance_h = temp_h / self.calibrated_frames
            self.distance_w = temp_w / self.calibrated_frames
            self.wharf_human_height = temp_wharf_human_height / self.calibrated_frames
    
    def clear_calibration_count(self):
        self.calibrated_frames = 0
        if hasattr(self, 'distance_w'):
            delattr(self, 'distance_w')
            delattr(self, 'distance_h')
            delattr(self, 'wharf_human_height')    
    
    def get_bboxes(self, tracker, names):
        boxes = []
        classes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            xyxy = track.to_tlbr()
            xywh = utills.xyxy2xywh(xyxy)
            class_num = track.class_num
            class_name = names[int(class_num)]
            boxes.append(xywh)
            classes.append(class_name)
        return boxes, classes
    
    def is_inside(self, point, box):
        x, y, w, h = box
        return point[0] >= x and point[0] <= x+w and point[1] >= y and point[1] <= y+h

    def preprocess_bboxes(self, boxes, classes):
        for i in range(len(boxes)):
            if classes[i] in ['suspended lean object', 'people', 'chain'] + self.ignored_classes:
                continue
            x, y, w, h = boxes[i]
            if y <= self.suspended_threshold:
                classes[i] = 'suspended lean object'
        if not 'suspended lean object' in classes:
            for i, cls in enumerate(classes):
                if cls == 'chain' and not 'chain' in self.ignored_classes:
                    x, y, w, h = boxes[i]
                    bottom1 = (x, y+h)
                    bottom2 = (x+w, y+h)
                    if y <= self.suspended_threshold:
                        not_inside = True
                        for i, box in enumerate(boxes):
                            if classes[i] != 'people':
                                if self.is_inside(bottom1, box) or self.is_inside(bottom2, box):
                                    not_inside = False
                        if not_inside:
                            classes[i] = 'suspended lean object'

    def find_transformed_centers(self, boxes, classes):
        centers = []
        for i in range(len(boxes)):
            if not classes[i] == 'suspended lean object':
                centers.append(None)
                continue
            center = (boxes[i][0] + boxes[i][2]/2, boxes[i][1] + boxes[i][3]/2)
            d1 = np.sqrt((center[0] - self.reference_points[3][0]) ** 2 + (center[1] - self.reference_points[3][1]))
            d2 = np.sqrt((center[0] - self.reference_points[2][0]) ** 2 + (center[1] - self.reference_points[2][1]))
            x = self.width * d1 / (d1 + d2)
            d3 = np.sqrt((center[0] - self.reference_points[1][0]) ** 2 + (center[1] - self.reference_points[1][1]))
            y = self.height * d2 / (d2 + d3)
            centers.append((x,y))
        return centers

    def calculate_distance(self, boxes, classes, distance_estimations, frame, count,ids):
        # boxes = []
        # classes = []
        #boxes, classes = self.get_bboxes(tracker, names)
        # self.preprocess_bboxes(boxes, classes)
        no_action=0
        roi_pts = np.array(self.reference_points, np.int32)
        cv2.polylines(frame, [roi_pts], True, (70, 70, 70), thickness=10)
        #frame1 = np.copy(frame)
        if hasattr(self, 'distance_w'):
            # print('Distance', len(boxes))
            pairs, warped_pts, danger_zones, heights = utills.get_distances(boxes, self.reference_points, self.perspective_transform, self.inverse_perspective_transform, classes, self.distance_w, self.distance_h, self.width, self.height, self.danger_zone_width_threshold, self.danger_zone_height_threshold, self.wharf_human_height, self.wharf)
            reversed_pts = utills.get_perspective_transform(warped_pts, self.inverse_perspective_transform)
            reversed_danger_zones = utills.get_reversed_danger_zones(danger_zones, self.inverse_perspective_transform)
            img = plot.draw_danger_zones(frame, reversed_danger_zones)
            img,self.all_violations = plot.social_distancing_view(img, pairs, boxes, reversed_pts, heights,ids,self.all_violations,count,self.fps,self.filename,self.wharf) #social_distancing_view(img, pairs, boxes, reversed_pts, heights)
            # print('Distance', len(boxes))
            if not self.wharf:
                roi_edge= self.edge_points
                #print(roi_edge)
                img,self.all_violations=plot.calculate_edge_to_person(roi_edge,img,boxes, classes,count,self.all_violations,ids,self.filename,self.fps)
                #print(self.all_violations)
                # Show/write image and videos
        else:
            no_action+=1
            print("no action {} and {}".format(no_action,count))
            img = plot.no_action(frame)
        if count != 0:
            self.output_movie.write(img)
        
        if count == 0:
            cv2.imwrite(f"{self.output_dir}/frame%d.jpg" % count, img)