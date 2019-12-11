import numpy as np
from training import rotate_sample, augment_images, split, map_images_to_3_channels, extract_coordinates_list_unzipped, scale_coordinates

from csv_reader import CsvReader
from img_reader import ImageReader

# file paths for training data
faces_file_path = 'face-images-with-marked-landmark-points/face_images.npz'
csv_file_path = 'face-images-with-marked-landmark-points/facial_keypoints.csv'

# helper classes
image_reader = ImageReader(faces_file_path)
csv_reader = CsvReader(csv_file_path)

image_data = image_reader.get_array()
image_data = image_data['face_images']
image_data = np.moveaxis(image_data, -1, 0)

# Must map all the images into 3-channels
image_data = map_images_to_3_channels(image_data)

# this is a dataframe
df = csv_reader.get_relevant_data()

# This gets all the coordinates from the dataframe
coordinates_list = extract_coordinates_list_unzipped(df)
coordinates_list = np.array(coordinates_list)
coordinates_list = scale_coordinates(coordinates_list)
# draw coordinates on image

# extracting and separating data as appropriate
X_train, y_train, X_test, y_test = split(image_data, coordinates_list)

augment_images(X_train, y_train)