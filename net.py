# implement some mad transfer learning here
import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications.vgg19 import VGG19

NUM_CHANNELS = 3
WIDTH = 96
HEIGHT = 96

# the output consists of 15 pairs of points
OUTPUT = 8

# transfer learning with tensorflow's vgg19 model trained on imagenet (1M+ images)
base_model = VGG19(
    input_shape=(HEIGHT, WIDTH, NUM_CHANNELS),
    include_top=False,
    weights='imagenet'
)

# freezing the weights so that we can leverage feature extraction
base_model.trainable = False

# adding our own 'top' of the network to fit our new output
output = base_model.output
output = layers.Flatten()(output)
output = layers.Dense(1024, activation='relu')(output)
output = layers.Dense(OUTPUT, activation='relu')(output)

model = models.Model(inputs=base_model.input, outputs=output)

print(model.summary())

def get_custom_model():
    base_model = VGG19(
        input_shape=(HEIGHT, WIDTH, NUM_CHANNELS),
        include_top=False,
        weights='imagenet'
    )

    output = base_model.get_layer('block3_pool').output
    output = layers.Flatten()(output)
    output = layers.Dense(256, activation='tanh')(output)
    output = layers.Dense(OUTPUT, activation='sigmoid')(output)

    model = models.Model(inputs=base_model.input, outputs=output)

    return model

# trains the vgg19 neural network
def train(model, X_train : np.ndarray, y_train : np.ndarray, X_test : np.ndarray, y_test : np.ndarray):

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    model_callback = [
        callbacks.ModelCheckpoint(
            filepath='modelMK2.h5',
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
