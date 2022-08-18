import cv2
import mmcv
import numpy as np
from shapely.geometry import Polygon
from mmsegmentation.mmseg.apis import inference_segmentor, init_segmentor
from scipy.spatial import distance as dist
import constants

class DetectWorkspace:
    def __init__(self):
        self.config = constants.SEGMENTATION_CONFIG_PATH
        self.checkpoint = constants.SEGMENTATION_MODEL_PATH
        self.model = init_segmentor(self.config, self.checkpoint, device="cuda:0")
    def detect_workspace(self, frame):
        frame_count = 0
        
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

