#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


###### This program was based on this example:
###### https://www.tensorflow.org/tutorials/images/classification


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Collect the dataset
    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    pathToZip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
    PATH = os.path.join(os.path.dirname(pathToZip), 'cats_and_dogs_filtered')

    # Vars for dataset paths
    trainDir = os.path.join(PATH, 'train')
    validationDir = os.path.join(PATH, 'validation')
    trainCatsDir = os.path.join(trainDir, 'cats')
    trainDogsDir = os.path.join(trainDir, 'dogs')
    validationCatsDir = os.path.join(validationDir, 'cats')
    validationDogsDir = os.path.join(validationDir, 'dogs')

    # Let's look at how many cats and dogs images are in the training and validation directory
    numCatsTr = len(os.listdir(trainCatsDir))
    numDogsTr = len(os.listdir(trainDogsDir))
    numCatsVal = len(os.listdir(validationCatsDir))
    numDogsVal = len(os.listdir(validationDogsDir))
    totalTrain = numCatsTr + numDogsTr
    totalVal = numCatsVal + numDogsVal

    print('Total training cat images:', numCatsTr)
    print('Total training dog images:', numDogsTr)
    print('Total validation cat images:', numCatsVal)
    print('Total validation dog images:', numDogsVal)
    print('---------------------------------')
    print('Total training images:', totalTrain)
    print('Total validation images:', totalVal)

    # For convenience, set up variables to use while pre-processing the dataset and training the network.
    batch_size = 128
    epochs = 15
    IMG_HEIGHT = 150
    IMG_WIDTH = 150

    # Data preparation
    # Scale to 0-1
    trainImageGenerator = ImageDataGenerator(rescale=1.0 / 255)
    validationImageGenerator = ImageDataGenerator(rescale=1.0 / 255)

    trainDataGen = trainImageGenerator.flow_from_directory(batch_size=batch_size,
                                                           directory=trainDir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
    valDataGen = validationImageGenerator.flow_from_directory(batch_size=batch_size,
                                                              directory=validationDir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

    sampleTrainingImages, _ = next(trainDataGen)
    # plotImages(sampleTrainingImages[:5])

    # Create the model
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.summary()

    # Create a checkpoint callback directory
    # https://www.tensorflow.org/tutorials/keras/save_and_load#save_checkpoints_during_training
    checkpointPath = 'trainingCheckpoints/cp.ckpt'
    checkpointDir = os.path.dirname(checkpointPath)

    # Create a callback that saves the model's weights
    cpCallback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPath,
                                                    save_weights_only=True,
                                                    verbose=1)

    # Train the model
    history = model.fit_generator(
        trainDataGen,
        steps_per_epoch=totalTrain,
        epochs=epochs,
        validation_data=valDataGen,
        validation_steps=totalVal,
        callbacks=[cpCallback]
    )

    # Visualize Results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# Save the final model
model.save('model.h5')
