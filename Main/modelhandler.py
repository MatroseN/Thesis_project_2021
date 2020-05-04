# Imports should be grouped in the following order:
#
# Standard library imports.
# Related third party imports.
# Local application/library specific imports.
# You should put a blank line between each group of imports.
# https://www.python.org/dev/peps/pep-0008/#imports

# System imports
import json
import pickle
import os
from os import path
from timeit import default_timer as timer

# 3rd party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from termcolor import colored
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Custom imports
import Main.Models as available_models

"""
This class performs following steps:
Alternative 1
    1. Trains the model
    2. Saves the model
    3. Logs training data
    4. Evaluates the model
    5. Logs evaluation data
Alternative 2
    1. Loads the saved model
    2. Evaluate the saved model
    3. Log evaluation data
"""


class ModelHandler:
    load_input_data_time = 0
    training_time = 0
    verbose = 0
    visualize = 0
    serial = -1
    limit = None

    def __init__(self, device, verbose, visualize):
        self.make_directories()
        self.data = None
        self.device = device
        self.random_seed = 29
        self.verbose = verbose
        self.visualize = visualize
        self.load_data()

    def load_data(self):
        """Load data (traffic signs)

        Load .pickle file form "Input/" containing the dataset (traffic signs)
        that the model are going to train on. The loading time is stored in
        the global variable "load_input_data_time".

        :raises
            IOError:  The dataset file you are trying to load from doesn't exist!
        """

        start_timer_loading = timer()

        try:
            with open('Input/data2.pickle', 'rb') as file:
                self.data = pickle.load(file, encoding='latin1')
        except IOError:
            print(colored("The dataset file you are trying to load from doesn't exist!", "red"))
            exit(0)

        # Making channels come at the end (3*32*32 ==> 32,*32*3)
        self.data['x_train'] = self.data['x_train'].transpose(0, 2, 3, 1)
        self.data['x_validation'] = self.data['x_validation'].transpose(0, 2, 3, 1)
        self.data['x_test'] = self.data['x_test'].transpose(0, 2, 3, 1)
        stop_timer_loading = timer()
        self.load_input_data_time = stop_timer_loading - start_timer_loading
        print("Dataset loading time: " + str(self.load_input_data_time) + "s")

    def predict(self, m, file_path, training_time):
        """Evaluate the trained model

        Evaluates the trained model by guessing what type of traffic sign
        the model receives. The result is stored in a .csv file "Custom_logs/<NAME>_prediction.csv".
        A confusion matrix is also generated and can be found in the .csv file
        "Custom_logs/<NAME>__confusion_matrix.csv"

        Args:
            :param m: The trained model
            :param file_path: Where the logs files are going to be stored at
            :param training_time: The total training time in seconds

        Returns:
            List containing the number of correct guesses and number
            of wrong guesses. [correct, incorrect]

        """
        results = m.model.predict(self.data['x_test'])

        correct = 0
        incorrect = 0
        # Declare and initialize a 2D confusion matrix
        matrix = [[0 for i in range(43)] for j in range(43)]
        for i in range(len(results)):
            if self.data['labels'][self.data['y_test'][i]] == self.data['labels'][np.argmax(results[i])]:
                correct += 1
                matrix[self.data['y_test'][i]][np.argmax(results[i])] += 1
                if self.visualize == 2 or self.visualize == 3:
                     self.visualize_prediction(i, self.data['labels'][np.argmax(results[i])])
            else:
                incorrect += 1
                matrix[self.data['y_test'][i]][np.argmax(results[i])] += 1
                if self.visualize == 1 or self.visualize == 3:
                    self.visualize_prediction(i, self.data['labels'][np.argmax(results[i])])

        # Save prediction data to CSV file
        prediction_data = pd.DataFrame(
            np.array([[
                m.get_name_with_timestamp(self.serial),
                str(correct),
                str((correct / (correct + incorrect))),
                str(incorrect),
                str((incorrect / (correct + incorrect))),
                str(training_time)
            ]]),
            columns=['Model', 'num_correct', 'num_correct_percent', 'num_incorrect', 'num_incorrect_percent', 'training_time'])
        prediction_data.to_csv(file_path + m.get_name_with_timestamp(self.serial) + "_prediction.csv")

        # Save confusion matrix data to CSV file
        matrix_data = pd.DataFrame(matrix, columns=self.data["labels"], index=self.data["labels"])
        matrix_data.to_csv(file_path + m.get_name_with_timestamp(self.serial) + "_confusion_matrix.csv")

        # Console print
        print("Correct: " + str(correct) + " (" + str(round((correct / (correct + incorrect)) * 100, 2)) + " %) "
                "\nIncorrect: " + str(incorrect) + " (" + str(round((incorrect / (correct + incorrect)) * 100, 2))
                + " %)\nTraining time: " + str(round(training_time, 2)) + "s")

        return [correct, incorrect]

    def train_model(self, model_to_train, model_path, shared_stats, x, y, z, serial, debug_mode):
        """Train the model

        This functions creates a folder for the trained model where all files
        that are related (settings, statistics etc) to the model are saved to. If the parameter
        "model_path" is NOT empty training will occur. If the parameter "model_path" IS empty
        evaluation/prediction will occur instead.

        There is a timer in this function that calculates the total training time. Total
        training time = "Save model settings" + "model training".

        Args:
            :param model_to_train: The specific model that is going to be trained
            :param model_path: Path to the model. If this value is empty evaluation/prediction will occur.
            :param shared_stats: Shared dictionary for statistics gathering
            :param x: placeholder for data
            :param y: placeholder for data
            :param z: placeholder for data
            :param serial: model serial number when training with grid search
            :param debug_mode: Limits the number of training samples (boolean)

        Raises:
            OSError: Creation of the directory %s failed
            IOError: Could not find that file. Check the spelling!
        """

        self.serial = serial

        if x == 0 and y == 0 and z == 0:
            m = model_to_train(self.random_seed)
        else:
            m = model_to_train(self.random_seed, x, y, z)

        # Create folder in Custom_logs for the model
        file_path = 'Custom_logs/' + m.get_name_with_timestamp(self.serial) + "/"
        try:
            os.mkdir(file_path)
        except OSError:
            print("Creation of the directory %s failed" % file_path)

        if len(model_path) <= 0:
            start_time_specific_model = 0
            stop_time_specific_model = 0
            statistics = []

            # Save model settings in to a json file.
            with open(file_path + m.get_name_with_timestamp(self.serial) + "_model_settings.json", 'w') as outfile:
                json.dump(json.loads(m.model.to_json()), outfile)

            with tf.device(self.device):
                start_time_specific_model = timer()
                tboard_log_dir = os.path.join("Logs", m.get_name_with_timestamp(self.serial))
                tensorboard = TensorBoard(
                    log_dir=tboard_log_dir,
                    histogram_freq=0.0,
                    write_graph=True,
                    write_images=True)

                early_stopping = EarlyStopping(
                    monitor='val_accuracy',
                    min_delta=0,
                    patience=15,
                    verbose=1,
                    mode='auto',
                    baseline=None,
                    restore_best_weights=True
                )

                save_callback = ModelCheckpoint(
                    filepath="Trained_Models/" + m.get_name_with_timestamp(self.serial) + "_epoch_{epoch}.h5",
                    monitor='val_accuracy',
                    verbose=m.verbose,
                    save_best_only=True,
                    save_weights_only=False,
                    mode='auto',
                    save_freq='epoch')

                if debug_mode:
                    self.limit = 10

                stats = m.model.fit(self.data['x_train'][:self.limit],
                                    self.data['y_train'][:self.limit],
                                    batch_size=m.batch_size,
                                    epochs=m.epochs,
                                    validation_data=(self.data['x_validation'], self.data['y_validation']),
                                    callbacks=[early_stopping, save_callback, tensorboard],
                                    verbose=self.verbose)

                # Save model hyper data data to CSV file
                hyper_data = []
                hyper, batch_size = m.get_variables()
                hyper_data.append("batch_size," + str(batch_size))
                for h in hyper:
                    hyper_data.append(h + "," + str(hyper[h]))
                hyper_df = pd.DataFrame(hyper_data, columns=["Hyper parameters"])
                hyper_df.to_csv(file_path + m.get_name_with_timestamp(self.serial) + "_hyper.csv")

                stop_time_specific_model = timer()
                statistics.append(m.name + " S" + str(serial))
                statistics.append(self.load_input_data_time)
                statistics.append(stop_time_specific_model - start_time_specific_model)

                df = pd.DataFrame(stats.history)
                df.to_csv(file_path + m.get_name_with_timestamp(self.serial) + ".csv")

                m.copy_model_file(serial, file_path)
                statistics.append(self.predict(m, file_path, (stop_time_specific_model - start_time_specific_model)))

            shared_stats[m.name + " S" + str(serial)] = statistics

        else:
            try:
                m.model.load_weights("Trained_Models/" + model_path + ".h5")
                self.predict(m, file_path, 0)
            except IOError:
                print(colored("Could not find that file. Check the spelling!", "red"))

    def make_directories(self):
        """ Create project folders

        Creates folders that are needed for the project. Reason why function exists is to ease the process
        when you clone the project from Git and make sure that the necessary folders exists.

         Raises:
            IOError: Could not create the folder!
        """
        directories = ["Custom_logs", "Input", "Logs", "Trained_models"]
        for directory in directories:

            if not path.isdir(directory):
                try:
                    os.mkdir(directory)
                except IOError:
                    print(colored("Could not create the folder!", "red"))

    def visualize_prediction(self, index, prediction):
        """ Display a image with the answer and prediction

        Displays a image on what the model thinks what the image represent and the actual answer.

        Args:
            :param index: Index location of the image
            :param prediction: What does the AI/Model think what the image represent
        """
        plt.grid(False)
        plt.imshow(self.data['x_test'][index])
        plt.title("Actual: " + self.data['labels'][self.data['y_test'][index]] + ", Index: " + str(index))
        plt.xlabel("Prediction: " + prediction)
        plt.show()

    def print_statistics(self, statistics):
        """Print statistics to the console

        This function only writes data to the console. With other words, this function does
        affect the Custom_logs files at all!

        Args:
            :param statistics: Shared dictionary with statistics data
        """
        total_training_time = 0
        # Calculate total training time
        for stat in statistics:
            total_training_time += statistics[stat][1] + statistics[stat][2]
        print("\nTotal training time: " + str(round(total_training_time, 2)) + "s")
        # Print individual training time for models in the order they were trained in
        print("\nSpecific model training time(s)\nName" + "\t\t\t\t" + "Time")
        for stat in statistics:
            tabs = self.decide_tabs(statistics[stat][0])
            print(str(statistics[stat][0]) + tabs + str(round(statistics[stat][2], 2)) + "s")
        # Print individual prediction accuracy for models in the order they were trained in
        print("\nModel prediction accuracy\nName" + "\t\t\t\t" + "Correct" + "\t\t\t\t" + "Incorrect")
        for stat in statistics:
            correct = statistics[stat][3][0]
            incorrect = statistics[stat][3][1]
            tabs = self.decide_tabs(statistics[stat][0])
            tabs1 = self.decide_tabs(str(correct) + " (" + str(round((correct / (correct + incorrect)) * 100, 2))
                                     + " %) ")
            print(str(statistics[stat][0]) + tabs + str(correct)
                  + " (" + str(round((correct / (correct + incorrect)) * 100, 2)) + " %) " + tabs1 + str(incorrect)
                  + " (" + str(round((incorrect / (correct + incorrect)) * 100, 2)) + " %) ")

    def prep_models(self, models_to_train):
        """Model preparation

        This function validates that every model that is entered in a driver actually exists.

        Args:
            :param models_to_train: List from a driver with models that we need to validate if they exists or not.

        :return: A list with validated models
        """
        models = []
        if len(models_to_train) <= 0:
            for model in available_models.models:
                if model is not None:
                    models.append(model)
        else:
            for model in models_to_train:
                if available_models.models[model] is not None:
                    models.append(available_models.models[model])
                else:
                    print(colored("A model you have chosen to train doesn't exist", "red"))
                    exit(0)

        return models

    def decide_tabs(self, length):
        """Decide how many tabs (\t) is needed

        This function does not affect any end result. This function is only used in the function
        "print_statistics" that print statistics to the console.

        :param length: String to be evaluated
        :return: How many tabs is needed for a better visual presentation in the console
        """

        if len(length) < 8:
            tabs = "\t\t\t\t"
        elif 7 < len(length) < 12:
            tabs = "\t\t\t"
        elif 11 < len(length) < 16:
            tabs = "\t\t"
        else:
            tabs = "\t"

        return tabs
