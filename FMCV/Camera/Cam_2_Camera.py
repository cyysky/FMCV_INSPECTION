from FMCV.Camera import Camera
from FMCV.Camera import Camera_2

import numpy as np

def get_image(name=None):
    frames_1 = Camera.get_image()
    frames_2 = Camera_2.get_image()
    combined_dict = {**frames_1, **frames_2}
    return combined_dict