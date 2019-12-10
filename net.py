# implement some mad transfer learning here

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