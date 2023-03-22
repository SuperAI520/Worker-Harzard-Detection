# imports
import cv2
import numpy as np
import constants
import math
from shapely.geometry import Polygon, MultiPolygon

UNIT_LENGTH = constants.UNIT_LENGTH

def get_bottom_points(boxes, classes):
    bottom_points = []
    for i, box in enumerate(boxes):
        if classes[i] in ['People']:
            pnts = [int(box[0]+(box[2]*0.5)),int(box[1]+box[3])]
            bottom_points.append(pnts)
        else:
            bottom_points.append(box)
    return bottom_points

def get_project_points(boxes, reference_points, classes, wharf):
    roi_bottom = max(reference_points[0][1], reference_points[1][1])
    roi_middle_y = (reference_points[0][1] + reference_points[1][1] + reference_points[2][1] + reference_points[3][1]) / 4
    roi_left = max(reference_points[0][0], reference_points[3][0])
    bottom_points = []
    for i, box in enumerate(boxes):
        if classes[i] == 'Suspended Lean Object':
            if not wharf:
                pnts = [int(box[0]+box[2]*0.5),int(box[1]*0.5 + box[3]*0.25 + roi_bottom*0.5)]
                bottom_points.append(pnts)
            else:
                pnts = [int(box[0]+box[2]*0.5),int(roi_middle_y)]
                pnts[0] = max(pnts[0], roi_left)
                bottom_points.append(pnts)
        else:
            bottom_points.append(box)
    return bottom_points

def get_project_points_wharf(boxes, land_Y, reference_points, classes):
    bottom_points = []
    for i, box in enumerate(boxes):
        if classes[i] == 'Suspended Lean Object':
                pnts = [int(box[0]+box[2]*0.5),land_Y]
                bottom_points.append(pnts)
        else:
            bottom_points.append(box)
    return bottom_points

def get_center_points(boxes):
    centers = []
    for box in boxes:
        pnts = [int(box[0]+box[2]*0.5),int(box[1] + box[3]*0.5)]
        centers.append(pnts)
    return centers

def get_perspective_transform(pts, transform):
    transformed_pts = []
    for pt in pts:
        pnts = np.array([[[int(pt[0]),int(pt[1])]]] , dtype="float32")
        bd_pnt = cv2.perspectiveTransform(pnts, transform)[0][0]
        pnt = [int(bd_pnt[0]), int(bd_pnt[1])]
        transformed_pts.append(pnt)
    return transformed_pts

def get_transformed_bbox(box, transform):
    x, y, w, h = box[0], box[1], box[2], box[3]
    left_top = [x, y]
    left_bottom = [x, y+h]
    right_top = [x+w, y]
    right_bottom = [x+w, y+h]
    return np.array(get_perspective_transform([left_bottom, right_bottom, right_top, left_top], transform))

def cal_dis(p1, p2, distance_w, distance_h):
    
    h = abs(p2[1]-p1[1])
    w = abs(p2[0]-p1[0])
    
    dis_w = float((w/distance_w)*UNIT_LENGTH)
    dis_h = float((h/distance_h)*UNIT_LENGTH)
    
    return int(np.sqrt(((dis_h)**2) + ((dis_w)**2)))

def is_inside_old(danger_zone, center, coord):
    inside_danger_zone = coord[0] >= danger_zone[0] and coord[0] <= danger_zone[2] \
                            and coord[1] >= danger_zone[1] and coord[1] <= danger_zone[3]
    return inside_danger_zone

def is_inside_old_wharf_alex(danger_zone, center, coord):
    if len(danger_zone) == 0:
        return False

    pts = np.array(danger_zone, np.int32)
    inside = cv2.pointPolygonTest(pts, coord, False)
    return inside >= 0

def get_reversed_danger_zones(danger_zones, transform):
    reversed_danger_zones = []
    for zone in danger_zones:
        left_top = np.array([[[int(zone[0]),int(zone[1])]]] , dtype="float32")
        left_bottom = np.array([[[int(zone[0]),int(zone[3])]]] , dtype="float32")
        right_top = np.array([[[int(zone[2]),int(zone[1])]]] , dtype="float32")
        right_bottom = np.array([[[int(zone[2]),int(zone[3])]]] , dtype="float32")
        pts = [left_top, left_bottom, right_bottom, right_top]
        zone = []
        for pt in pts:
            reversed = cv2.perspectiveTransform(pt, transform)[0][0]
            zone.append([int(reversed[0]), int(reversed[1])])
        reversed_danger_zones.append(zone)
    return reversed_danger_zones

def get_hatch_reference_points(hatch_reference):
    points = []
    points.append(hatch_reference[0])
    points.append((hatch_reference[0][0] - hatch_reference[1][0] // 2, hatch_reference[0][1] - hatch_reference[1][1] // 2))
    points.append((hatch_reference[0][0] + hatch_reference[1][0] // 2, hatch_reference[0][1] - hatch_reference[1][1] // 2))
    points.append((hatch_reference[0][0] + hatch_reference[1][0] // 2, hatch_reference[0][1] + hatch_reference[1][1] // 2))
    points.append((hatch_reference[0][0] - hatch_reference[1][0] // 2, hatch_reference[0][1] + hatch_reference[1][1] // 2))
    return points

def get_distances(boxes, reference_points, hatch_reference, perspective_transform, inverse_perspective_transform, classes, old_classes, distance_w, distance_h, w, h, danger_zone_height_threshold, wharf):
    danger_zones = []
    danger_zone_checks = []
    suspended_cargo_ids = []
    transformed_hatch_reference_points = []
    roi_middle_y = (reference_points[0][1] + reference_points[1][1] + reference_points[2][1] + reference_points[3][1]) / 4
    bottom_points = get_bottom_points(boxes, classes)
    bottom_points = get_project_points(bottom_points, reference_points, classes, wharf)
    bottom_points = get_perspective_transform(bottom_points, perspective_transform)
    center_points = get_center_points(boxes)

    if len(hatch_reference) != 0:
        hatch_reference_points = get_hatch_reference_points(hatch_reference)
        transformed_hatch_reference_points = get_perspective_transform(hatch_reference_points, perspective_transform)
    transformed_center_points = get_perspective_transform(center_points, perspective_transform)
    height_from_ground = [-1] * len(center_points)
    for i, cls in enumerate(classes):
        if cls == 'Suspended Lean Object':
            bottom_points[i][1] = max(h/2, bottom_points[i][1])
            bottom_points[i][1] = min(h, bottom_points[i][1])
            bottom_points[i][0] = max(0, bottom_points[i][0])
            bottom_points[i][0] = min(w, bottom_points[i][0])
            
            height_from_ground[i] = cal_dis(bottom_points[i], transformed_center_points[i], distance_w, distance_h)
            suspended_cargo_ids.append(i)

    for i in range(len(bottom_points)):
        if classes[i] != 'Suspended Lean Object':
            continue
        danger_zone = None
        if height_from_ground[i] >= danger_zone_height_threshold:
            transformed_box = get_transformed_bbox(boxes[i], perspective_transform)
            danger_zone = calculate_danger_zone_coordinates(old_classes[i], boxes[i], transformed_box, bottom_points[i], reference_points, transformed_hatch_reference_points, distance_w, distance_h, w, h, wharf)
            danger_zones.append(danger_zone)
        for j in range(len(bottom_points)):
            if classes[j] != 'People':
                continue
            in_danger = is_inside_old(danger_zone, bottom_points[i], bottom_points[j]) if height_from_ground[i] >= danger_zone_height_threshold else False
            danger_zone_checks.append((i, j, None, in_danger)) 
    return suspended_cargo_ids, danger_zone_checks, bottom_points, danger_zones, height_from_ground

def get_danger_zones_wharf(boxes, wharf_landing_Y, wharf_person_height_thr, reference_points, classes, old_classes, w, h, danger_zone_height_threshold):
    danger_zones = []
    danger_zone_checks = []
    suspended_cargo_ids = []
    bottom_points = get_bottom_points(boxes, classes)
    bottom_points = get_project_points_wharf(bottom_points, wharf_landing_Y, reference_points, classes)
    center_points = get_center_points(boxes)
    height_from_ground = [-1] * len(center_points)
    for i, cls in enumerate(classes):
        if cls == 'Suspended Lean Object':
            bottom_points[i][1] = max(h/2, bottom_points[i][1])
            bottom_points[i][1] = min(h, bottom_points[i][1])
            bottom_points[i][0] = max(0, bottom_points[i][0])
            bottom_points[i][0] = min(w, bottom_points[i][0])
            height_from_ground[i] = wharf_landing_Y - (boxes[i][1] + boxes[i][3])

    for i in range(len(bottom_points)):
        if classes[i] != 'Suspended Lean Object':
            continue
        danger_zone = None
        if height_from_ground[i] >= danger_zone_height_threshold:
            center_x = boxes[i][0] + boxes[i][2] / 2
            # if boxes[i][2] < (w // 3):
            #     left_top = [center_x - boxes[i][2] * 3 / 4, wharf_landing_Y - boxes[i][3] / 4]
            #     left_bottom = [center_x - boxes[i][2], wharf_landing_Y + boxes[i][3] / 4]
            #     right_top = [center_x + boxes[i][2] * 3 / 4, wharf_landing_Y - boxes[i][3] / 4]
            #     right_bottom = [center_x + boxes[i][2], wharf_landing_Y + boxes[i][3] / 4]
            # else:
            left_top = [center_x - boxes[i][2] / 2, wharf_landing_Y - boxes[i][3] / 4]
            left_bottom = [center_x - boxes[i][2] * 3 / 4, wharf_landing_Y + boxes[i][3] / 4]
            right_top = [center_x + boxes[i][2] / 2, wharf_landing_Y - boxes[i][3] / 4]
            right_bottom = [center_x + boxes[i][2] * 3 / 4, wharf_landing_Y + boxes[i][3] / 4]

            # danger_zone = [left_top, left_bottom, right_bottom, right_top]
            danger_poly = Polygon([left_top, left_bottom, right_bottom, right_top])
            workarea_poly = Polygon(reference_points.squeeze(axis=1))
            danger_zone = danger_poly.intersection(workarea_poly)
            if danger_zone.geom_type == 'MultiPolygon':
                danger_zone =list(danger_zone.geoms)[0]
            # elif danger_zone.geom_type == 'Polygon':
            #     print('Polygon')
            int_coords = lambda x: np.array(x).round().astype(np.int32)
            danger_zone = int_coords(danger_zone.exterior.coords)
            danger_zones.append(danger_zone)
        suspended_cargo_ids.append(i)
        for j in range(len(bottom_points)):
            if classes[j] != 'People':
                continue
            if boxes[j][3] < wharf_person_height_thr:
                continue
            in_danger = is_inside_old_wharf_alex(danger_zone, bottom_points[i], bottom_points[j]) if height_from_ground[i] >= danger_zone_height_threshold else False
            danger_zone_checks.append((i, j, None, in_danger)) 

    return suspended_cargo_ids, danger_zone_checks, bottom_points, danger_zones, height_from_ground

def calculate_danger_zone_coordinates(old_class, box, transformed_box, center_pt, reference_points, hatch_reference_points, distance_w, distance_h, w, h, wharf):
    if len(hatch_reference_points) != 0:
        box_w = max(math.dist(transformed_box[0], transformed_box[1]), math.dist(transformed_box[2], transformed_box[3]))
        box_h = max(math.dist(transformed_box[0], transformed_box[3]), math.dist(transformed_box[2], transformed_box[1]))
        box_diag = np.sqrt(box_w ** 2 + box_h ** 2)
        distance1 = math.dist(hatch_reference_points[1], hatch_reference_points[3])
        distance2 = math.dist(hatch_reference_points[2], hatch_reference_points[4])
        cargo_len = (distance1 + distance2)/2
        out_rate = cargo_len / box_diag
        # if out_rate > 1:
        #     out_rate = 1
        transformed_w = out_rate * box_w
        transformed_h = out_rate * box_h
        # transformed_w = transformed_w * 1.2
        # transformed_h = transformed_h * 1.2
        left = max(0, center_pt[0] - transformed_w/2)
        right = min(w, center_pt[0] + transformed_w/2)
        top = max(0, hatch_reference_points[0][1] - transformed_h/2)
        bottom = min(h, hatch_reference_points[0][1] + transformed_h/2)
    else:
        box_w, box_h = box[2:]
        box_diag = np.sqrt(box_w ** 2 + box_h ** 2)
        distance1 = np.linalg.norm(np.cross(reference_points[1]-reference_points[0],reference_points[2]-reference_points[0])/np.linalg.norm(reference_points[1]-reference_points[0]))
        distance2 = np.linalg.norm(np.cross(reference_points[1]-reference_points[0],reference_points[3]-reference_points[0])/np.linalg.norm(reference_points[1]-reference_points[0]))
        min_height = min(distance1, distance2)

        # transform_rate = h / box_diag
        transformed_w = box_w
        if box_diag > min_height:
            if old_class in ['Container', 'Small Pipe', 'Large Pipe', 'Suspended Lean Object', 'Wooden Board', 'Iron Rake', 'Wood', 'Steel Plate']:
                transformed_h = np.sqrt(h ** 2 - transformed_w ** 2)
            else:
                transformed_h = np.sqrt((h / 2) ** 2 - transformed_w ** 2)
        else:
            transformed_h = (h / min_height) * box_h
        # transformed_w = transformed_w * 1.2
        # transformed_h = transformed_h * 1.2
        left = max(0, center_pt[0] - transformed_w/2)
        right = min(w, center_pt[0] + transformed_w/2)
        top = max(0, center_pt[1] - transformed_h/2)
        bottom = min(h, center_pt[1] + transformed_h/2)
        return [left, top, right, bottom]

    return [left, top, right, bottom]