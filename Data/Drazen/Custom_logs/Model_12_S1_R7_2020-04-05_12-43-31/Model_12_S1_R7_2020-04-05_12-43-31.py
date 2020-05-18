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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D


""" 
This is a dummy for speed testing
"""


class Model_12:
    def __init__(self, random_seed, serial, seed_serial, lr, momentum, batch_size):
        self.author = "Drazen"
        self.name = self.__class__.__name__
        self.description = "Batch size"
        self.serial = serial
        self.ss = seed_serial
        self.seed = random_seed

        # Setting up seed for repeatability
        # More info on https://github.com/NVIDIA/tensorflow-determinism
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        rn.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        self.timestamp = self.get_timestamp()
        self.epochs = 50
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
        optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    def get_name_with_timestamp(self):
        return self.name + "_S" + str(self.serial) + "_R" + str(self.ss) + "_" + self.timestamp

    def get_timestamp(self):
        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        return str(formatted_timestamp)

    def copy_model_file(self, file_path):
        shutil.copy2("Models/" + self.name + ".py", file_path + self.get_name_with_timestamp() + ".py")

    def get_variables(self):
        return self.model.optimizer.__dict__, self.batch_size, self.seed
