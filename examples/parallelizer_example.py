"""
Name: parallizer.py
Author: Ronald Kemker
Description: Parallelize Keras Models

Note:
Requires Tensorflow, Keras
https://www.tensorflow.org/
https://keras.io/

Tested on Ubuntu 16.04, Anaconda Python 3.6, 4x NVIDIA Titan X GPUs
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
from machine_learning.parallelizer import Parallelizer

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, -1)
x_test = x_test.reshape(10000, -1)
model = Sequential()
model.add(Dense(64, input_shape=(784,), activation='relu'))
model.add(Dense(10, activation='softmax'))

para = Parallelizer()
parallel_model = para.transform(model)

batch_size = 128 * para.n_gpus

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

parallel_model.compile(optimizer='nadam', loss='categorical_crossentropy',
                       metrics=['accuracy'])

parallel_model.fit(x_train, y_train, batch_size=batch_size,
                   validation_data=(x_test, y_test))