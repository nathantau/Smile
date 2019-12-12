import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import ModelCheckpoint

NUM_CHANNELS = 3
WIDTH = 96
HEIGHT = 96
# the output consists of 4 pairs of points
OUTPUT = 8

def get_model_details(model):
    print(model.summary())

def get_custom_model_w_transfer():
    '''
    Retrieves a custom neural network based off of VGG19.
    '''
    
    base_model = VGG19(
        input_shape=(HEIGHT, WIDTH, NUM_CHANNELS),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    output = base_model.get_layer('block1_pool').output
    output = layers.Flatten()(output)
    output = layers.Dense(256, activation='tanh')(output)
    output = layers.Dense(OUTPUT, activation='sigmoid')(output)

    model = models.Model(inputs=base_model.input, outputs=output)

    return model

def get_custom_model():
    '''
    Retrieves a custom neural network.
    '''

    model = models.Sequential()

    model.add(layers.Conv2D(8, (3,3), activation='sigmoid', input_shape=(96,96,3)))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(16, (3,3), activation='sigmoid'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(16, (3,3), activation='sigmoid'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='sigmoid'))
    model.add(layers.Dense(OUTPUT, activation='sigmoid'))

    return model

def train(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    '''
    Consumes and trains a neural network passed into it.
    '''
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    model_callback = [
        ModelCheckpoint(
            filepath='modelMKIII.h5',
            save_best_only=True,
            monitor='val_loss',
            verbose=1)
    ]

    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=model_callback,
    )

    return history
