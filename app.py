import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import cv2
import argparse

from prediction import get_trained_model, predict_image, predict_images
from img_reader import ImageReader
from csv_reader import CsvReader
from training import train, split
from net import get_custom_model

def main():

    # Handle arguments from CMD
    argument_parser = argparse.ArgumentParser(description='Choose whether you want to train network or test network')
    argument_parser.add_argument('config', help='choose whether to train or test network')
    args = argument_parser.parse_args().config

    print('the inputted args is', args)

    if args is None:
        print('This configuration is unsupported. Please enter [train] or [test]')

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
    # -----------------------------------------
    # image = image_data[1]
    # image = convert_to_3_channels(image)
    # print(image.shape)
    # plt.imshow((image * 255).astype(np.uint8))
    # plt.show()
    # -----------------------------------------

    # this is a dataframe
    df = csv_reader.get_relevant_data()
 
    # This gets all the coordinates from the dataframe
    coordinates_list = extract_coordinates_list_unzipped(df)
    coordinates_list = np.array(coordinates_list)
    coordinates_list = scale_coordinates(coordinates_list)
    # draw coordinates on image
    
    # validation to ensure data is clean
    validate_data(image_data, coordinates_list)

    # extracting and separating data as appropriate
    X_train, y_train, X_test, y_test = split(image_data, coordinates_list)

    if args == 'train':
        print('[train] configuration selected. Training.')
        train(get_custom_model(), X_train, y_train, X_test, y_test)
    elif args == 'test':
        print('[test] configuration selected. Testing.')
        model = get_trained_model()
        prediction = predict_images(model, X_test)
        assert len(X_test) == len(prediction)
        draw_images(X_test, prediction)    
    else:
        print('This configuration is unsupported. Please enter [train] or [test]')
    
# Converts list of coordinates into coordinate pairs
def zip_to_coordinate_pairs(coordinates : np.ndarray):
    iterable_coordinates = iter(coordinates)
    zipped_list = list(zip(iterable_coordinates, iterable_coordinates))
    zipped_list = np.array(zipped_list)
    return zipped_list

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

    # dropping out invalid data
    coordinates_list = [[coordinate if not np.isnan(coordinate) else 0 for coordinate in coordinates] for coordinates in coordinates_list]

    num_zeroes = 0
    for coordinates in coordinates_list:
        for coordinate in coordinates:
            if coordinate == 0:
                num_zeroes += 1

    return coordinates_list

# draws on each of the images
def draw_images(images : np.ndarray, coordinates_list : np.ndarray):
    coordinates_list = scale_coordinates(coordinates_list, scale=96.0)
    for image, coordinates in zip(images, coordinates_list):
        coordinates = zip_to_coordinate_pairs(coordinates)
        draw(image, coordinates)
        # print(image.shape)


# draws landmarks for one image using coordinates
def draw(image, coordinates):
    print('coordinate pairs are', coordinates)
    plt.imshow((image * 255).astype(np.uint8))
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

def scale_coordinates(coordinates_list : np.ndarray, scale=float(1.0/96.0)):
    scaled_coordinates_list = np.array([coordinates * scale for coordinates in coordinates_list])
    return scaled_coordinates_list

def validate_data(images : np.ndarray, coordinates_list : np.ndarray):
    assert not np.any(np.isnan(images))
    assert not np.any(np.isnan(coordinates_list))

if __name__ == '__main__':
    main()
    