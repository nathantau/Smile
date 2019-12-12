import tensorflow.compat.v1 as tf
from tensorflow.keras.models import load_model 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import cv2

def get_trained_model(filename: str):
    model = load_model(filename)
    return model

def predict_images(model, test_images: np.ndarray):
    return np.array(model.predict(test_images))

def predict_image(model, test_image: np.ndarray):
    test_image = np.reshape(test_image, (1,96,96,3))
    output = model.predict(test_image)
    output = np.array(output)
    return output



