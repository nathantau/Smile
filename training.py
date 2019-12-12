import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt
import math

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TRAIN_PERCENTAGE = 0.8
IMG_SIZE = 96

'''
This is a utility class with many functions to allow for easy training/data manipulation
'''

def rotate_sample(image: np.ndarray, coordinates: np.ndarray):
    '''
    Rotates a given image and its respective facial coordinates
    by either 90, 180, 270, or 360 degrees. It returns a tuple of 
    the augmented images and coordinates.
    '''
    rotations = random.randint(0, 3)
    coordinates = scale_coordinates(coordinates, scale=96.0)
    iterable_coordinates = iter(coordinates)
    zipped_coordinates = zip(iterable_coordinates, iterable_coordinates)

    for _ in range(rotations):
        # this is 1 rotation of 90 degrees counterclockwise
        image = np.rot90(image)
        zipped_coordinates = [(coordinate_pair[1], IMG_SIZE - coordinate_pair[0]) for coordinate_pair in zipped_coordinates]

    unzipped_coordinates = list(sum(zipped_coordinates, ()))
    unzipped_coordinates = np.array(unzipped_coordinates)
    unzipped_coordinates = scale_coordinates(unzipped_coordinates, scale=1.0/96.0)

    return image, unzipped_coordinates

def augment_images(images: np.ndarray, coordinates_list: np.ndarray, generate_data=True):
    '''
    Consumes a list of images and a list of coordinates and returns a list of modified
    images and corresponding coordinates. This leverages the rotate_sample(...) method and 
    takes in an optional parameter to determine if data should be generated and saved
    in *npy files.
    '''
    for _ in range(1):
        count = 0
        total_length = len(images)

        new_images = []
        new_coordinates_list = []

        for image, coordinates in zip(images, coordinates_list):
            print('sample number:', count, '/', total_length)
            image, coordinates = rotate_sample(image, coordinates)

            new_images.append(image)
            new_coordinates_list.append(coordinates)

            if generate_data:
                count += 1

    np.save('images.npy', new_images)
    np.save('coordinates_list.npy', new_coordinates_list)


# Used to generate and save training data to save time 
def generate_and_save_training_data(images: np.ndarray, coordinates_list: np.ndarray):
    augment_images(images, coordinates_list)

def split(images: np.ndarray , coordinates_list: np.ndarray):
    '''
    Consumes a list of images and a list of coordinates (1 set of
    facial features) and splits them into training sets and testing
    sets in the form of (X_train, y_train, X_test, y_test)
    '''
    train_size = int(len(images) * TRAIN_PERCENTAGE)
    test_size = len(images) - train_size
    
    print('training size', train_size)
    print('validation size', test_size)

    # use np functions to split data
    X_train = images[:train_size]
    y_train = coordinates_list[:train_size]
    X_test = images[train_size:]
    y_test = coordinates_list[train_size:]

    print('training X size', len(X_train), 'training y size', len(y_train))
    print('testing X size', len(X_test), 'testing y size', len(y_test))

    assert train_size == len(X_train) and train_size == len(y_train)
    assert test_size == len(X_test) and test_size == len(y_test)

    return X_train, y_train, X_test, y_test 

def convert_to_3_channels(image: np.ndarray):
    '''
    Given a single-channel image, it converts it 
    into 3 channels for neural network compatibility
    '''
    image = cv2.merge((image, image, image))
    return image

def map_images_to_3_channels(images: np.ndarray):
    '''
    Consumes a list of 1-channel images and maps them to 
    a list of 3 channel images, by layering each image with
    copies of itself.
    '''
    converted_images = [convert_to_3_channels(image) for image in images]
    print('converted images are of type', type(converted_images))
    converted_images = np.array(converted_images)

    print('sample shape', converted_images[0].shape)

    return converted_images

def extract_coordinates_list_unzipped(df):
    '''
    Consumes a pandas dataframe consisting of the coordinates 
    of facial features and returns a list of list of coordinates
    '''
    coordinates_list = []
    for _, rows in df.iterrows():
        unzipped_list = [row for row in rows]
        coordinates_list.append(list(unzipped_list))

    # dropping out invalid data
    coordinates_list = [[coordinate if not np.isnan(coordinate) else 0 for coordinate in coordinates] for coordinates in coordinates_list]
    
    return coordinates_list

def extract_coordinates_list(df):
    '''
    Consumes a pandas dataframe consisting of the coordinates 
    of facial features and returns a list of list of coordinate pairs
    '''
    coordinates_list = []
    
    for _, rows in df.iterrows():
        unzipped_list = [row for row in rows]
        iterable_list = iter(unzipped_list)
        zipped_list = zip(iterable_list, iterable_list)
        coordinates_list.append(list(zipped_list))

    return coordinates_list

def scale_coordinates_list(coordinates_list: np.ndarray, scale=float(1.0/96.0)):
    '''
    Consumes a list of list of coordinate pairs and returns
    the same list with each coordinate scaled by a given amount
    '''
    scaled_coordinates_list = np.array([coordinates * scale for coordinates in coordinates_list])

    return scaled_coordinates_list

def scale_coordinates(coordinates: np.ndarray, scale=float(1.0/96.0)):
    '''
    Consumes a list of coordinate pairs and returns
    the same list with each coordinate scaled by a given amount
    '''
    scaled_coordinates = np.array([coordinate * scale for coordinate in coordinates])

    return scaled_coordinates
