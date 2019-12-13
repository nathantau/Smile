import cv2
import numpy as np
from prediction import get_trained_model

class Streamer():

    def __init__(self):
        self.model = get_trained_model('modelMKIII.h5')

    def stream(self):
        # Do the prediction here, maybe import drawing methods