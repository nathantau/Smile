import numpy as np
import tensorflow.compat.v1 as tf
import os
from tensorflow.keras import models, layers, callbacks

from net import model

TRAIN_PERCENTAGE = 0.8

def split(images : np.ndarray , coordinates_list : np.ndarray):

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

# trains the vgg19 neural network
def train(model, X_train : np.ndarray, y_train : np.ndarray, X_test : np.ndarray, y_test : np.ndarray):

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    model_callback = [
        callbacks.ModelCheckpoint(
            filepath='model_{epoch}.h5',
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
