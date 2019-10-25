#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


if __name__ == '__main__':
    # Use these to load a model from an h5 file
    model = tf.keras.models.load_model('model.h5')
    model.summary()

    # Use this to run the model on new data
    # https://www.tensorflow.org/tutorials/load_data/csv
    predictions = model.predict(test_data)
    print(predictions)
