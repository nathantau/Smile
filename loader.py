import numpy as np
import math
from matplotlib.patches import Rectangle

import pandas as pd
import matplotlib.pyplot as plt
from training import rotate_sample, scale_coordinates

class Loader():
    '''
    This class is used for loading the training data stored in *.npy formats.
    '''
    def __init__(self):
        super()

    def load_from_filepath(self, filepath: str):
        print('Loading data from filepath {}'.format(filepath))
        np_array = np.load(filepath)
        return np_array
