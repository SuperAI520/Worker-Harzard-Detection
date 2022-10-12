from fastai.vision.all import *
import constants

def get_keypoints(frame):
    calib_points = constants.REFERENCE_AREA
    return calib_points