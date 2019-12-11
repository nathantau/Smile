import numpy as np
import os
import random
import cv2

TRAIN_PERCENTAGE = 0.8
IMG_SIZE = 96

# To augment images and avoid overfitting, we add in some rotations
def rotate_sample(image : np.ndarray, coordinates : np.ndarray):

    rotations = random.randint(0, 4)
    iterable_coordinates = iter(coordinates)
    zipped_coordinates = zip(iterable_coordinates, iterable_coordinates)

    for _ in range(rotations):
        # this is a rotation of 90 degrees counterclockwise
        image = np.rot90(image)
        # handle updated coordinates
        zipped_coordinates = [(coordinate_pair[1], IMG_SIZE - coordinate_pair[0]) for coordinate_pair in zipped_coordinates]

    # unzipped_coordinates = [item for tuple in zipped_coordinates for item in tuple]
    unzipped_coordinates = list(sum(zipped_coordinates, ()))
    unzipped_coordinates = np.array(unzipped_coordinates)

    return image, unzipped_coordinates

# Used for lengthenening dataset with augmentations
def augment_images(images : np.ndarray, coordinates_list : np.ndarray):
    for _ in range(1):
        count = 0
        total_length = len(images)
        for image, coordinates in zip(images, coordinates_list):
            print('sample number:', count, '/', total_length)
            image, coordinates = rotate_sample(image, coordinates)
            images = np.append(images, image)
            coordinates_list = np.append(coordinates_list, coordinates)
            count += 1
            if count % 50 == 0:
                # np.savez('training_data2.npz', images=images, coordinates_list=coordinates_list)
                np.save('images.npy', images)
                np.save('coordinates_list.npy', coordinates_list)

    return images, coordinates_list

# Used to generate and save training data to save time 
def generate_and_save_training_data(images : np.ndarray, coordinates_list : np.ndarray):
    images, coordinates_list = augment_images(images, coordinates_list)
    # np.savez('training_data2.npz', images=images, coordinates_list=coordinates_list)
    return 'training_data.npz'

# Separates data into X_train, y_train, X_test, y_test
def split(images : np.ndarray , coordinates_list : np.ndarray):
    # images, coordinates_list = augment_images(images, coordinates_list)

    train_size = int(len(images) * TRAIN_PERCENTAGE)
    test_size = len(images) - train_size
    
    print('training size', train_size)
    print('validation size', test_size)

    # use np functions to split data
    X_train = images[:train_size]
    y_train = coordinates_list[:train_size]

    X_test = images[train_size:]
    y_test = coordinates_list[train_size:]

    print('training size', len(X_train))
    print('test size', len(X_test))

    assert train_size == len(X_train) and train_size == len(y_train)
    assert test_size == len(X_test) and test_size == len(y_test)

    return X_train, y_train, X_test, y_test 

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

# gets all the unzipped coordinates from dataframe
def extract_coordinates_list_unzipped(df):
    coordinates_list = []
    
    for index, rows in df.iterrows():
        unzipped_list = [row for row in rows]
        coordinates_list.append(list(unzipped_list))

    # dropping out invalid data
    coordinates_list = [[coordinate if not np.isnan(coordinate) else 0 for coordinate in coordinates] for coordinates in coordinates_list]

    num_zeroes = 0
    for coordinates in coordinates_list:
        for coordinate in coordinates:
            if coordinate == 0:
                num_zeroes += 1

    return coordinates_list

def scale_coordinates(coordinates_list : np.ndarray, scale=float(1.0/96.0)):
    scaled_coordinates_list = np.array([coordinates * scale for coordinates in coordinates_list])
    return scaled_coordinates_list
