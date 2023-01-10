import cv2
import mmcv
import numpy as np
from scipy.spatial import distance as dist
import constants
import math
import collections 
from loguru import logger

TRACK_LIMIT_TIME = 0.5
MOVEMENT_THR = 50
TRACKING_CYCLE = 3
TOLERANCE_DIS = -50

class CargoTracker:
    def __init__(self, wharf, fps, h, w):
        self.wharf = wharf
        self.ignore_cargo_ids = []
        self.accumulated_cargo_ids = []
        self.distances_array = []
        self.cargos_pos = []
        self.fps = fps
        self.height = h
        self.width = w
        self.step = 0 # 0: not process 1: ready to detect first cargo detection 2: in process 3: end process
        self.wharf_landing_Y = []
        self.hatch_reference_pos = []
        self.tracking_cycles = TRACKING_CYCLE

    def clear(self):
        self.distances_array.clear()
        self.cargos_pos.clear()
        self.accumulated_cargo_ids.clear()
        self.ignore_cargo_ids.clear()
        self.step = 1
        self.wharf_landing_Y.clear()
        # self.hatch_reference_pos.clear()
        self.tracking_cycles = TRACKING_CYCLE

    
    def set_step(self, step):
        self.step = step

    def track_no_detection_case(self, work_area_index, workspace_contours):
        if self.step == 1:
            self.step == 2

        if self.step == 2:
            delete_idxs = []
            min_index = -1
            for n in range(len(self.accumulated_cargo_ids)):
                if len(self.cargos_pos[n]) < self.fps * TRACK_LIMIT_TIME: # At least 3 seconds must be tracked cargo.
                    #delete_idxs.append(n)
                    continue

                init_pos, last_pos = self.cargos_pos[n][0], self.cargos_pos[n][-1]
                # print(last_pos, init_pos)
                diff_pos = (last_pos[0] - init_pos[0], last_pos[1] - init_pos[1])
                if diff_pos[1] > 0 or abs(diff_pos[1]) < MOVEMENT_THR: # Process only unloading cargos and there should be a Y-axis movement.
                    delete_idxs.append(n)
                    continue

                min_value = min(self.distances_array[n])
                min_index = self.distances_array[n].index(min_value)
                work_area_index = min_index
                logger.debug(f'********** 1 ***********   unloaded cargo {self.accumulated_cargo_ids[n]} started from {min_index}st workarea')
                flag = False
                for z, iid_y in enumerate(self.hatch_reference_pos):
                    if iid_y[0] == self.accumulated_cargo_ids[n]:
                        flag = True
                        break
                if not flag:                                    
                    inside = cv2.pointPolygonTest(workspace_contours[work_area_index], init_pos['pos'], True)
                    if inside >= 0 :
                        self.hatch_reference_pos.append([self.accumulated_cargo_ids[n], (init_pos['pos'][0], init_pos['pos'][1] + init_pos['size'][1] // 4), init_pos['size']])
                        self.tracking_cycles -= 1
                if self.tracking_cycles == 0:
                    self.step = 3

                return True, work_area_index

        return False, work_area_index

    def get_wharf_landing_Y(self):
        sum_Y = 0
        if len(self.wharf_landing_Y) == 0:
            return 0

        for pair in self.wharf_landing_Y:
            sum_Y += pair[1]
        landing_Y = int(math.floor(sum_Y / len(self.wharf_landing_Y)))
        return landing_Y

    def get_hatch_reference(self):
        reference_pos = []
        cnt = len(self.hatch_reference_pos)
        if cnt == 0:
            return []
        center_pt = [0, 0]
        init_size = [0, 0]
        for pair in self.hatch_reference_pos:
            center_pt[0] += pair[1][0]
            center_pt[1] += pair[1][1]
            init_size[0] += pair[2][0]
            init_size[1] += pair[2][1]
        reference_pos = [(center_pt[0] // cnt, center_pt[1] // cnt), (init_size[0] // cnt, init_size[1] // cnt)]
        return reference_pos


    def track(self, work_area_index, ids, classes, bboxes, center_points, workspaces, workspace_contours):
        success = False
        if self.step == 1:
            self.step = 2
            if 'Suspended Lean Object' in classes:
                for k in range(len(classes)):
                    if classes[k] == 'Suspended Lean Object':
                        self.ignore_cargo_ids.append(ids[k])

        if self.step == 2:
            current_cargo_ids = []
            for k in range(len(classes)):
                if classes[k] == 'Suspended Lean Object':
                    if ids[k] in self.ignore_cargo_ids:
                        continue

                    current_cargo_ids.append(ids[k])
                    if not self.wharf:
                        center_point = (int(bboxes[k][0] + bboxes[k][2] / 2), int(bboxes[k][1] + bboxes[k][3] / 2))
                    else:
                        center_point = (int(bboxes[k][0] + bboxes[k][2] / 2), int(bboxes[k][1] + bboxes[k][3]))

                    distances = []
                    if not self.wharf:
                        for pt in center_points:
                            dist = math.hypot(center_point[0]-pt[0], center_point[1]-pt[1])
                            distances.append(int(dist))
                    else:
                        # dist = math.hypot(0, center_point[1]-0)
                        # distances.append(int(dist))
                        distances.append(bboxes[k][3]) # save height of cargo rect in wharf case

                    for n in range(len(self.accumulated_cargo_ids)):
                        if ids[k] == self.accumulated_cargo_ids[n]:
                            self.distances_array[n].append(distances)
                            cargo_pos = {'pos': center_point, 'size': (int(bboxes[k][2]), int(bboxes[k][3]))}
                            self.cargos_pos[n].append(cargo_pos)

                    if not ids[k] in self.accumulated_cargo_ids:
                        if self.wharf and len(workspaces) > 0: # ignore new cargo inside the ground from tracking.
                            inside = cv2.pointPolygonTest(workspaces[0], center_point, False)
                            if inside >= 0:
                                self.ignore_cargo_ids.append(ids[k])
                                continue

                        self.accumulated_cargo_ids.append(ids[k])
                        self.distances_array.append([distances])
                        cargo_pos = {'pos': center_point, 'size': (int(bboxes[k][2]), int(bboxes[k][3]))}
                        self.cargos_pos.append([cargo_pos])

            if collections.Counter(self.accumulated_cargo_ids) != collections.Counter(current_cargo_ids):
                delete_idxs = []
                for n in range(len(self.accumulated_cargo_ids)):
                    if self.accumulated_cargo_ids[n] not in current_cargo_ids:
                        # print(f'--------->>   Processing {self.accumulated_cargo_ids[n]} movement')
                        if len(self.cargos_pos[n]) < self.fps * TRACK_LIMIT_TIME: # At least 3 seconds must be tracked cargo.
                            # print(f'XXXXXXXX  invalid short movement   id: {self.accumulated_cargo_ids[n]}')
                            # delete_idxs.append(n)
                            continue

                        init_pos, last_pos = self.cargos_pos[n][0], self.cargos_pos[n][-1]
                        # print(last_pos, init_pos)
                        diff_pos = (last_pos['pos'][0] - init_pos['pos'][0], last_pos['pos'][1] - init_pos['pos'][1])
                        if not self.wharf:
                            if diff_pos[1] > 0 or abs(diff_pos[1]) < MOVEMENT_THR: # Process only unloading cargos and there should be a Y-axis movement.
                                logger.debug(f'XXXXXXXX  invalid unloading movement   id: {self.accumulated_cargo_ids[n]}')
                                delete_idxs.append(n)
                                continue

                            min_value = min(self.distances_array[n][0])
                            min_index = self.distances_array[n][0].index(min_value)
                            work_area_index = min_index
                            success = True
                            flag = False
                            for z, iid_y in enumerate(self.hatch_reference_pos):
                                if iid_y[0] == self.accumulated_cargo_ids[n]:
                                    flag = True
                                    break
                            if not flag:                                    
                                inside = cv2.pointPolygonTest(workspace_contours[work_area_index], init_pos['pos'], True)
                                if inside >= 0 :
                                    self.hatch_reference_pos.append([self.accumulated_cargo_ids[n], (init_pos['pos'][0], init_pos['pos'][1] + init_pos['size'][1] // 4), init_pos['size']])
                                    self.tracking_cycles -= 1
                            if self.tracking_cycles == 0:
                                self.step = 3
                            
                            logger.debug(f'********** 2 ***********   unloaded cargo {self.accumulated_cargo_ids[n]} started from {min_index}st workarea')
                        else:
                            if diff_pos[1] < 0 or abs(diff_pos[1]) < 50: # Process only unloading cargos and there should be a Y-axis movement.
                                logger.debug(f'XXXXXXXX  invalid loading movement   id: {self.accumulated_cargo_ids[n]}')
                                delete_idxs.append(n)
                                continue
                            offset = 0
                            TOLERANCE_DIS = (-1) * self.distances_array[n][-1][0] / 2
                            if len(workspaces) > 0:
                                inside = cv2.pointPolygonTest(workspaces[0], last_pos['pos'], True)
                                # if inside < TOLERANCE_DIS:
                                #     print(f'$$$$$$$$$$$$$$$$$$$$$$$$$   out of workspace {inside}')
                                #     delete_idxs.append(n)
                                #     continue

                                offset = (abs(inside) + self.distances_array[n][-1][0] / 4) if inside < 0 else 0 
                            logger.debug(f'********** 2 ***********   loaded cargo {self.accumulated_cargo_ids[n]} to ({last_pos})')
                            
                            flag = False
                            for z, iid_y in enumerate(self.wharf_landing_Y):
                                if iid_y[0] == self.accumulated_cargo_ids[n]:
                                    self.wharf_landing_Y[z][1] = last_pos['pos'][1] + offset
                                    flag = True
                                    break

                            if not flag:
                                self.wharf_landing_Y.append([self.accumulated_cargo_ids[n], last_pos['pos'][1] + offset])
                                self.tracking_cycles -= 1
                            
                            if self.tracking_cycles == 0:
                                logger.debug('###########################   END tracking')
                                self.step = 3

                        self.distances_array.clear()
                        self.cargos_pos.clear()
                        self.accumulated_cargo_ids.clear()
                        self.ignore_cargo_ids.clear()
                        break
                            
                        
                        
                # print(f'delete ids {delete_idxs}')
                self.accumulated_cargo_ids = [c for j, c in enumerate(self.accumulated_cargo_ids) if j not in delete_idxs]
                self.cargos_pos = [c for j, c in enumerate(self.cargos_pos) if j not in delete_idxs]
                self.distances_array = [c for j, c in enumerate(self.distances_array) if j not in delete_idxs]

            # print(self.accumulated_cargo_ids, current_cargo_ids, self.wharf_landing_Y)
            
        return success, work_area_index
