import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from training import scale_coordinates

def zip_to_coordinate_pairs(coordinates: np.ndarray):
    '''
    Consumes a list of coordinates (per 1 image) and converts
    it into coordinate pairs.
    '''
    iterable_coordinates = iter(coordinates)
    zipped_list = list(zip(iterable_coordinates, iterable_coordinates))
    zipped_list = np.array(zipped_list)
    return zipped_list

def draw_images(images: np.ndarray, coordinates_list: np.ndarray):
    '''
    Consumes a list of images and a list of coordinates, performing
    the draw() operation on each of them.
    '''
    for image, coordinates in zip(images, coordinates_list):
        draw(image, coordinates)


def draw(image: np.ndarray, coordinates: np.ndarray):
    '''
    Consumes an image and coordinate pairs matplotlib to draw coordinates 
    on a given image.
    '''
    coordinates = scale_coordinates(coordinates, scale=96.0)
    coordinates = zip_to_coordinate_pairs(coordinates)
    print('coordinate pairs are', coordinates)
    plt.imshow((image * 255).astype(np.uint8))
    num = 0
    for coordinate_pair in coordinates:
        color = choose_color(num)
        if not math.isnan(coordinate_pair[0]) and not math.isnan(coordinate_pair[1]): 
            plt.gca().add_patch(Rectangle((coordinate_pair[0],coordinate_pair[1]),1,1, color=color))
        num += 1

    plt.show()

def choose_color(num: int):
    '''
    Utility method to determine which color should be plotted
    on an image.
    '''
    if num == 0:
        color = 'red'
    elif num == 1:
        color = 'blue'
    elif num == 2:
        color = 'green'
    elif num == 3:
        color = 'white'
    return color
