import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import cv2
import argparse

from prediction import get_trained_model, predict_image, predict_images
from img_reader import ImageReader
from csv_reader import CsvReader
from training import split, rotate_sample, generate_and_save_training_data, map_images_to_3_channels, extract_coordinates_list_unzipped, extract_coordinates_list, scale_coordinates
from loader import Loader
from drawer import draw_images, draw

from net import get_custom_model, train, get_custom_model_w_transfer

def main():

    '''
    This is the entrypoint of the whole application. From the CMD,
    the user can specify either 'train' or 'test' after 'python app.py'
    to choose the mode which they would like to execute.
    '''
    # handle arguments from CMD
    argument_parser = argparse.ArgumentParser(description='Choose whether you want to train network or test network')
    argument_parser.add_argument('config', help='choose whether to train or test network')
    args = argument_parser.parse_args().config

    print('the inputted args is', args)

    if args is None:
        print('This configuration is unsupported. Please enter [train] or [test]')
        return

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
    df = csv_reader.get_facial_features()
 
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

        # image_reader = ImageReader('training_data.npz')
        # data = image_reader.get_array()
        # X_train = data['images']
        # y_train = data['coordinates_list']

        loader = Loader()

        X_train = loader.load_from_filepath('images.npy')
        y_train = loader.load_from_filepath('coordinates_list.npy')
        
        print('images size', len(X_train))
        print('coordinates size', len(y_train))

        assert len(X_train) == len(y_train)

        train(get_custom_model(), X_train, y_train, X_test, y_test)

    elif args == 'test':
        print('[test] configuration selected. Testing.')
        model = get_trained_model('modelMKIII.h5')
        prediction = predict_images(model, X_test)
        assert len(X_test) == len(prediction)
        draw_images(X_test, prediction)

    else:
        print('This configuration is unsupported. Please enter [train] or [test]')


def check_image_shapes(images: np.ndarray):
    '''
    Utility method to check that all images have the correct shape.
    '''
    for image in images:
        if image.shape is not (96,96):
            print(image.shape)

def validate_data(images: np.ndarray, coordinates_list: np.ndarray):
    '''
    Utility method to verify that data is sanitized
    '''
    assert not np.any(np.isnan(images))
    assert not np.any(np.isnan(coordinates_list))

if __name__ == '__main__':
    main()
    