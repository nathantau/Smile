import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import cv2

from img_reader import ImageReader
from csv_reader import CsvReader
from training import train, split
from net import model

def main():

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

    # Sample image
    # image = image_data[1]
    # image = convert_to_3_channels(image)
    # print(image.shape)
    # plt.imshow((image * 255).astype(np.uint8))
    # plt.show()

    # this is a dataframe
    df = csv_reader.get_relevant_data()
 
    # This gets all the coordinates from the dataframe
    coordinates_list = extract_coordinates_list_unzipped(df)
    coordinates_list = np.array(coordinates_list)


    # draw coordinates on image
    



    
    X_train, y_train, X_test, y_test = split(image_data, coordinates_list)


    print(y_train[0])
    # print(y_train[0].shape)

    train(model, X_train, y_train, X_test, y_test)





# gets all the coordinates from dataframe
def extract_coordinates_list(df):
    coordinates_list = []
    
    for index, rows in df.iterrows():
        unzipped_list = [row for row in rows]
        iterable_list = iter(unzipped_list)
        zipped_list = zip(iterable_list, iterable_list)
        coordinates_list.append(list(zipped_list))

    return coordinates_list

# gets all the unzipped coordinates from dataframe
def extract_coordinates_list_unzipped(df):
    coordinates_list = []
    
    for index, rows in df.iterrows():
        unzipped_list = [row for row in rows]
        coordinates_list.append(list(unzipped_list))

    return coordinates_list

# draws landmarks for one image using coordinates
def draw(image, coordinates):

    plt.imshow(image)

    for coordinate_pair in coordinates:
        if not math.isnan(coordinate_pair[0]) and not math.isnan(coordinate_pair[1]): 
            plt.gca().add_patch(Rectangle((coordinate_pair[0],coordinate_pair[1]),1,1, color='red'))

    plt.show()

# checks to see that all images have the same input size
def check_image_shapes(images):
    for image in images:
        if image.shape is not (96,96):
            print(image.shape)

# converts image to 3 channels
def convert_to_3_channels(image):
    image = cv2.merge((image, image, image))
    return image

# maps each image in a series of images into 3 channels
def map_images_to_3_channels(images : np.ndarray):
    converted_images = [convert_to_3_channels(image) for image in images]
    print('converted images are of type', type(converted_images))
    converted_images = np.array(converted_images)

    print('sample shape', converted_images[0].shape)

    return converted_images


if __name__ == '__main__':
    main()
    