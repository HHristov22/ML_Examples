# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.__version__

from tensorflow.keras.models import load_model
import os

# Define the model path
model_path = './DeepLearning/ConvolutionalNeuralNetworks(CNN)/cnn_model.h5'

# Part 1 - Data Preprocessing
# dataset -> https://www.kaggle.com/datasets/pushpakhinglaspure/cats-vs-dogs/data

dataset_path = rf'/home/jesus/LocalDisk/dogs_vs_cats/'
# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory(rf'{dataset_path}/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(rf'{dataset_path}/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Load CNN model / Building the CNN
# Try to load the CNN model
if os.path.exists(model_path):
    print("Loading the existing model...")
    cnn = load_model(model_path)
else:
    print("No saved model found. Building and training a new model...")
# Initialising the CNN
    cnn = tf.keras.models.Sequential()

    # Step 1 - Convolution
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

    # Step 2 - Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Adding a second convolutional layer
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Step 3 - Flattening
    cnn.add(tf.keras.layers.Flatten())

    # Step 4 - Full Connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

    # Step 5 - Output Layer
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Part 3 - Training the CNN

    # Compiling the CNN
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Training the CNN on the Training set and evaluating it on the Test set
    cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

    # Save the trained model
    cnn.save('./DeepLearning/ConvolutionalNeuralNetworks(CNN)/cnn_model.h5')

# Part 4 - Making a single prediction

import numpy as np
from keras.preprocessing import image

for index_img in range(1, 6):
    test_image = image.load_img(rf'{dataset_path}/single_prediction/proba{index_img}.jpg', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    print(training_set.class_indices)
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print(prediction, "\n\n")