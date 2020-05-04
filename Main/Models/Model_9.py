# Imports should be grouped in the following order:
#
# Standard library imports.
# Related third party imports.
# Local application/library specific imports.
# You should put a blank line between each group of imports.
# https://www.python.org/dev/peps/pep-0008/#imports

# System imports
import os
import random as rn
import shutil
from datetime import datetime

# 3rd party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D
from tensorflow.keras.models import Sequential


""" 
This is a dummy for speed testing
"""


class Model_9:
    def __init__(self, random_seed, lr, momentum, batch_size):
        self.author = "Drazen"
        self.name = self.__class__.__name__
        self.description = "Dummy for testing"

        # Setting up seed for repeatability
        # More info on https://github.com/NVIDIA/tensorflow-determinism
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        rn.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

        self.timestamp = self.get_timestamp()
        self.epochs = 1
        self.batch_size = batch_size
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

    def get_name_with_timestamp(self, serial):
        return self.name + "_S" + str(serial) + "_" + self.timestamp

    def get_timestamp(self):
        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        return str(formatted_timestamp)

    def copy_model_file(self, serial, file_path):
        shutil.copy2("Models/" + self.name + ".py", file_path + self.get_name_with_timestamp(serial) + ".py")

    def get_variables(self):
        return self.model.optimizer.__dict__, self.batch_size
