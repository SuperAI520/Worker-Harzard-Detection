import cv2
import mmcv
import numpy as np
from shapely.geometry import Polygon
from mmsegmentation.mmseg.apis import inference_segmentor, init_segmentor
from scipy.spatial import distance as dist
import constants
import math

class DetectWorkspace:
    def __init__(self):
        self.config = constants.SEGMENTATION_CONFIG_PATH
        self.checkpoint = constants.SEGMENTATION_MODEL_PATH
        self.model = init_segmentor(self.config, self.checkpoint, device="cuda:0")
    def detect_workspace(self, frame):
        frame_count = 0
        direction = 0
        
        workspaces = []
        height_edges = []
        if frame is None:
            return None, None
        
        result = inference_segmentor(self.model, frame)
        seg = result[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        min_size = int(min(seg.shape[0], seg.shape[1]) / 10)
        color_seg[seg == 1, :] = [255, 0, 0]
        # mmcv.imwrite(color_seg, "out.png")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_size,min_size))
        color_seg = cv2.morphologyEx(color_seg, cv2.MORPH_OPEN, kernel)	
        color_seg = cv2.cvtColor(color_seg, cv2.COLOR_BGR2GRAY)
        cnts, hiers = cv2.findContours(color_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area_max = 0
        #Find biggest contour
        biggest_cnt = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if (area_max < area):
                area_max = area
                biggest_cnt = cnt 

        hull = cv2.convexHull(biggest_cnt)
        
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [hull], 0, 255, -1) # Draw filled contour in mask
        out = np.zeros_like(gray_image) # Extract out the object and place into output image
        out[mask == 255] = gray_image[mask == 255]
        ordered_box = order_points(box)
        blur = cv2.GaussianBlur(out, (5,5), 1)
        blur = blur.astype(np.float32)
        blur /=255.0
        kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float32)
        kernely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float32)
        gradient_x = cv2.filter2D(blur, -1, kernelx)
        gradient_y = cv2.filter2D(blur, -1, kernely)
        gradient_magnitude = np.hypot(gradient_x, gradient_y)
        dx = gradient_x*255.0
        dy = gradient_y*255.0
        dxx = np.interp(dx, (0, dx.max()), (0, 255))
        dyy = np.interp(dy, (0, dy.max()), (0, 255))
        _, dxx_thr = cv2.threshold(dxx, 50, 255, cv2.THRESH_TOZERO)
        _, dyy_thr = cv2.threshold(dyy, 50, 255, cv2.THRESH_TOZERO)
        x_sum = dxx_thr.sum()
        y_sum = dyy_thr.sum()
        print(x_sum, y_sum)
        if y_sum / x_sum >= 1.5:
            direction = 1

        print(f'direction {direction}')
        orientation = cv2.phase(dx, dy, angleInDegrees=True)
        thresh = 100
        _, binary_image = cv2.threshold(gradient_magnitude*255.0, thresh, 255, cv2.THRESH_BINARY)
        unit_degree = 360 / 20
        numbers_of_degree = []
        for i in range(20):
            n = np.sum((binary_image == 255) & (orientation >= unit_degree * i) & (orientation < unit_degree * (i + 1)))
            numbers_of_degree.append([i, n])
        
        arr_numbers_of_degree = np.array(numbers_of_degree)
        arr_numbers_of_degree = arr_numbers_of_degree[arr_numbers_of_degree[:, 1].argsort()[::-1][:20]] 
        real_angle_arr = []
        last_number = arr_numbers_of_degree[0][1]
        
        s0_1 = abs((ordered_box[2][1] - ordered_box[3][1])/(ordered_box[2][0] - ordered_box[3][0])) if ordered_box[2][0] - ordered_box[3][0] != 0 else -1
        s0_2 = abs((ordered_box[0][1] - ordered_box[3][1])/(ordered_box[0][0] - ordered_box[3][0])) if ordered_box[0][0] - ordered_box[3][0] != 0 else -1

        if direction == 0:    
            s0 = min(s0_1, s0_2) if s0_1 != -1 and s0_2 != -1 else max(s0_1, s0_2)
        else:
            s0 = max(s0_1, s0_2) if s0_1 != -1 and s0_2 != -1 else -1

        total_num = np.sum(arr_numbers_of_degree[:, 1])
        limit_num = total_num / 5

        for i in range(20):
            angle_idx = arr_numbers_of_degree[i][0]
            angle_count = arr_numbers_of_degree[i][1]
            
            if (total_num < limit_num or last_number / angle_count >= 2) and len(real_angle_arr) >= 2:
                break
            total_num -= angle_count
            last_number = angle_count
            real_angle = angle_idx * unit_degree + 9 - 90

            if real_angle < 0:
                real_angle += 180
            if real_angle >= 180:
                real_angle -= 180

            s1 = np.tan(np.radians(real_angle))
            if s0 != -1:
                ang = abs(angle(s1, s0))
            else:
                ang = abs(real_angle - 90)

            if ang < 30:
                continue

            if direction == 1:
                real_angle = angle_idx * unit_degree + 9

            real_angle_arr.append(real_angle)

            if direction == 1 and len(real_angle_arr) >= 2:
                break

        real_angle_arr.sort()
        count_real_angle_arr = len(real_angle_arr)

        if direction == 0:
            s1 = np.tan(np.radians(real_angle_arr[count_real_angle_arr - 1]))
            s2 = np.tan(np.radians(real_angle_arr[0]))

            if s0 == s0_1:
                x = ordered_box[3][0] + 10 # top left
                y = s1*(x - ordered_box[3][0]) + ordered_box[3][1]
                x1, y1 = line_intersection(((x, y), ordered_box[3]), (ordered_box[0], ordered_box[1]))
                x = ordered_box[2][0] + 10 # top left
                y = s2*(x - ordered_box[2][0]) + ordered_box[2][1]
                x2, y2 = line_intersection(((x, y), ordered_box[2]), (ordered_box[0], ordered_box[1]))
                ordered_box[0] = (x1, y1)
                ordered_box[1] = (x2, y2)
            else:
                x = ordered_box[0][0] + 10 # top left
                y = s1*(x - ordered_box[0][0]) + ordered_box[0][1]
                x1, y1 = line_intersection(((x, y), ordered_box[0]), (ordered_box[1], ordered_box[2]))
                x = ordered_box[3][0] + 10 # top left
                y = s2*(x - ordered_box[3][0]) + ordered_box[3][1]
                x2, y2 = line_intersection(((x, y), ordered_box[3]), (ordered_box[1], ordered_box[2]))
                ordered_box[1] = (x1, y1)
                ordered_box[2] = (x2, y2)
            ordered_box = order_points(ordered_box)
            # print(x1, y1)
            # print(x2, y2)
        else:
            max_angle = real_angle_arr[count_real_angle_arr - 1] - 90
            min_angle = real_angle_arr[0] - 90
            max_angle = max_angle if max_angle >= 0 else max_angle + 180
            min_angle = min_angle if min_angle >= 0 else min_angle + 180
            max_angle = max_angle if max_angle >= 180 else max_angle - 180
            min_angle = min_angle if min_angle >= 180 else min_angle - 180
            s1 = np.tan(np.radians(max_angle))
            s2 = np.tan(np.radians(min_angle))
            ordered_box1 = ordered_box.copy()
            ordered_box2 = ordered_box.copy()
            if s0_2 == -1 or s0 == s0_2:
                x = ordered_box[3][0] + 10 # top left
                y = s2*(x - ordered_box[3][0]) + ordered_box[3][1]
                x1, y1 = line_intersection(((x, y), ordered_box[3]), (ordered_box[1], ordered_box[2]))
                x = ordered_box[0][0] + 10 # top left
                y = s1*(x - ordered_box[0][0]) + ordered_box[0][1]
                x2, y2 = line_intersection(((x, y), ordered_box[0]), (ordered_box[1], ordered_box[2]))
                ordered_box1[2] = (x1, y1)
                ordered_box1[1] = (x2, y2)
            else:
                x = ordered_box[2][0] + 10 # top left
                y = s2*(x - ordered_box[2][0]) + ordered_box[2][1]
                x1, y1 = line_intersection(((x, y), ordered_box[2]), (ordered_box[1], ordered_box[0]))
                x = ordered_box[3][0] + 10 # top left
                y = s1*(x - ordered_box[3][0]) + ordered_box[3][1]
                x2, y2 = line_intersection(((x, y), ordered_box[3]), (ordered_box[1], ordered_box[0]))
                ordered_box1[1] = (x1, y1)
                ordered_box1[0] = (x2, y2)
            ordered_box1 = order_points(ordered_box1)

            if s0_2 == -1 or s0 == s0_2:
                x = ordered_box[3][0] + 10 # top left
                y = s1*(x - ordered_box[3][0]) + ordered_box[3][1]
                x1, y1 = line_intersection(((x, y), ordered_box[3]), (ordered_box[1], ordered_box[0]))
                x = ordered_box[2][0] + 10 # top left
                y = s2*(x - ordered_box[2][0]) + ordered_box[2][1]
                x2, y2 = line_intersection(((x, y), ordered_box[2]), (ordered_box[1], ordered_box[0]))
                ordered_box2[0] = (x1, y1)
                ordered_box2[1] = (x2, y2)
            else:
                x = ordered_box[3][0] + 10 # top left
                y = s2*(x - ordered_box[3][0]) + ordered_box[3][1]
                x1, y1 = line_intersection(((x, y), ordered_box[3]), (ordered_box[1], ordered_box[2]))
                x = ordered_box[0][0] + 10 # top left
                y = s1*(x - ordered_box[0][0]) + ordered_box[0][1]
                x2, y2 = line_intersection(((x, y), ordered_box[0]), (ordered_box[1], ordered_box[2]))
                ordered_box2[2] = (x1, y1)
                ordered_box2[1] = (x2, y2)
            ordered_box2 = order_points(ordered_box2)

            polygon0 = Polygon(hull.squeeze()) # hull
            polygon1 = Polygon(ordered_box1) # approx
            polygon2 = Polygon(ordered_box2) # rotated_rect

            intersect1 = polygon1.intersection(polygon0).area
            union1 = polygon1.union(polygon0).area
            iou1 = intersect1 / union1

            intersect2 = polygon2.intersection(polygon0).area
            union2 = polygon2.union(polygon0).area
            iou2 = intersect2 / union2

            if iou1 > iou2:
                ordered_box = ordered_box1
            else:
                ordered_box = ordered_box2
            # print(f'  iou1 > iou2 {iou1 > iou2}')

        workspaces.append(ordered_box)

        
        """
        peri = cv2.arcLength(hull, True)
        num_corner = 100
        rate = 0.01
        while num_corner > 4:
            approx = cv2.approxPolyDP(hull, rate * peri, True)
            num_corner = len(approx)
            rate += 0.01

        polygon0 = Polygon(hull.squeeze()) # hull
        polygon1 = Polygon(approx.squeeze()) # approx
        polygon2 = Polygon(box) # rotated_rect

        intersect1 = polygon1.intersection(polygon0).area
        union1 = polygon1.union(polygon0).area
        iou1 = intersect1 / union1

        intersect2 = polygon2.intersection(polygon0).area
        union2 = polygon2.union(polygon0).area
        iou2 = intersect2 / union2
        
        
        if iou1 > iou2:
            new_approx = approx.squeeze()
            ordered_approx = order_points(new_approx)
            workspaces.append(ordered_approx)
        else:
            ordered_box = order_points(box)
            workspaces.append(ordered_box)
        # workspaces.append(workspace)
        """




        edge_pts = []
        for cnt in cnts:
            num_of_edge = len(cnt)
            approx = cnt
            rate = 0.0001
            while num_of_edge > 50:
                epsilon = rate*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                num_of_edge = len(approx)
                rate += 0.0001
            cnt = approx
            edge_pts.append(cnt)
            cv2.drawContours(color_seg, [cnt], -1, (0, 0, 255), 2)

        height_edges = edge_pts
        return workspaces, height_edges


def order_points(pts):
    	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([bl, br, tr, tl], dtype="float32")

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

def slope(x1, y1, x2, y2): # Line slope given two points:
    return (y2-y1)/(x2-x1)

def angle(s1, s2): 
    return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))