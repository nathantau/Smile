import numpy as np

class ImageReader():

    def __init__(self, npz_path):
        self.array = np.load(npz_path, mmap_mode='r')

    def get_array(self):
        return self.array