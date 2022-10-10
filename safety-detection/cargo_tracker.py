import cv2
import mmcv
import numpy as np
from scipy.spatial import distance as dist
import constants
import math
import collections 

TRACK_LIMIT_TIME = 1.5
MOVEMENT_THR = 50

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

    def clear(self):
        self.distances_array.clear()
        self.cargos_pos.clear()
        self.accumulated_cargo_ids.clear()
        self.ignore_cargo_ids.clear()
        self.step = 1
    
    def set_step(self, step):
        self.step = step

    def track_no_detection_case(self, work_area_index):
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
                print(last_pos, init_pos)
                diff_pos = (last_pos[0] - init_pos[0], last_pos[1] - init_pos[1])
                if diff_pos[1] > 0 or abs(diff_pos[1]) < MOVEMENT_THR: # Process only unloading cargos and there should be a Y-axis movement.
                    delete_idxs.append(n)
                    continue

                min_value = min(self.distances_array[n])
                min_index = self.distances_array[n].index(min_value)
                work_area_index = min_index
                print(f'********** 1 ***********   unloaded cargo {self.accumulated_cargo_ids[n]} started from {min_index}st workarea')
                self.step = 3
                return True, work_area_index

        return False, work_area_index

    def track(self, work_area_index, ids, classes, bboxes, center_points):
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
                    center_point = (int(bboxes[k][0] + bboxes[k][2] / 2), int(bboxes[k][1] + bboxes[k][3] / 2))
                    distances = []
                    for pt in center_points:
                        dist = math.hypot(center_point[0]-pt[0], center_point[1]-pt[1])
                        distances.append(int(dist))

                    for n in range(len(self.accumulated_cargo_ids)):
                        if ids[k] == self.accumulated_cargo_ids[n]:
                            self.distances_array[n].append(distances)
                            self.cargos_pos[n].append(center_point)

                    if not ids[k] in self.accumulated_cargo_ids:
                        self.accumulated_cargo_ids.append(ids[k])
                        self.distances_array.append([distances])
                        self.cargos_pos.append([center_point])

            if collections.Counter(self.accumulated_cargo_ids) != collections.Counter(current_cargo_ids):
                delete_idxs = []
                for n in range(len(self.accumulated_cargo_ids)):
                    if self.accumulated_cargo_ids[n] not in current_cargo_ids:
                        print(f'--------->>   Processing {self.accumulated_cargo_ids[n]} movement')
                        if len(self.cargos_pos[n]) < self.fps * TRACK_LIMIT_TIME: # At least 3 seconds must be tracked cargo.
                            print(f'XXXXXXXX  invalid short movement   id: {self.accumulated_cargo_ids[n]}')
                            # delete_idxs.append(n)
                            continue

                        init_pos, last_pos = self.cargos_pos[n][0], self.cargos_pos[n][-1]
                        print(last_pos, init_pos)
                        diff_pos = (last_pos[0] - init_pos[0], last_pos[1] - init_pos[1])
                        if diff_pos[1] > 0 or abs(diff_pos[1]) < MOVEMENT_THR: # Process only unloading cargos and there should be a Y-axis movement.
                            print(f'XXXXXXXX  invalid unloading movement   id: {self.accumulated_cargo_ids[n]}')
                            delete_idxs.append(n)
                            continue

                        min_value = min(self.distances_array[n][0])
                        print(self.distances_array[n][0])
                        min_index = self.distances_array[n][0].index(min_value)
                        work_area_index = min_index
                        success = True
                        
                        print(f'********** 2 ***********   unloaded cargo {self.accumulated_cargo_ids[n]} started from {min_index}st workarea')
                        print(center_points)
                        self.step = 3
                        
                print(f'delete ids {delete_idxs}')
                self.accumulated_cargo_ids = [c for j, c in enumerate(self.accumulated_cargo_ids) if j not in delete_idxs]
                self.cargos_pos = [c for j, c in enumerate(self.cargos_pos) if j not in delete_idxs]
                self.distances_array = [c for j, c in enumerate(self.distances_array) if j not in delete_idxs]

            print(self.accumulated_cargo_ids, current_cargo_ids)
            
        return success, work_area_index