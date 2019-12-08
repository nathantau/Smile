import numpy as np
import matplotlib.pyplot as plt

from img_reader import ImageReader
from csv_reader import CsvReader

def main():

    faces_file_path = 'face-images-with-marked-landmark-points/face_images.npz'
    csv_file_path = 'face-images-with-marked-landmark-points/facial_keypoints.csv'

    image_reader = ImageReader(faces_file_path)

    image_data = image_reader.get_array()
    image_data = image_data['face_images']
    image_data = np.moveaxis(image_data, -1, 0)

    # Sample image
    image = image_data[0]

    # plt.imshow(image)
    # plt.show()

    csv_reader = CsvReader(csv_file_path)

    print(csv_reader.get_data())
    



if __name__ == '__main__':
    main()
    