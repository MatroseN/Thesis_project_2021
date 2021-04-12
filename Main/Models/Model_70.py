# Imports should be grouped in the following order:
#
# Standard library imports.
# Related third party imports.
# Local application/library specific imports.
# You should put a blank line between each group of imports.
# https://www.python.org/dev/peps/pep-0008/#imports

# System imports
import os
import shutil
from datetime import datetime

# 3rd party imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D


""" 
This is for Bayesian optimization on learning rate and momentum
"""


class Model_70:
    def __init__(self, lr, momentum):
        self.author = "Felix"
        self.name = self.__class__.__name__
        self.description = "Learning rate and momentum for Bayesian"

        # Setting up seed for repeatability
        # More info on https://github.com/NVIDIA/tensorflow-determinism
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

        self.timestamp = self.get_timestamp()
        self.epochs = 50
        self.batch_size = 32
        self.verbose = 1

        self.model = Sequential()
        self.model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(AvgPool2D())
        self.model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        self.model.add(AvgPool2D())
        self.model.add(Flatten())
        self.model.add(Dense(units=120, activation='relu'))
        self.model.add(Dense(units=84, activation='relu'))
        self.model.add(Dense(units=43, activation='softmax'))
        optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=momentum)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    def get_name_with_timestamp(self):
        return self.name + self.timestamp

    def get_timestamp(self):
        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        return str(formatted_timestamp)

    def copy_model_file(self, file_path):
        shutil.copy2("Models/" + self.name + ".py", file_path + self.get_name_with_timestamp() + ".py")

    def get_variables(self):
        return self.model.optimizer.__dict__, self.batch_size, self.seed
