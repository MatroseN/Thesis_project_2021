import csv
import pickle


# Defining function for getting texts for every class - labels
from termcolor import colored


def label_text(file):
    # Defining list for saving label in order from 0 to 42
    label_list = []

    # Opening 'csv' file and getting image's labels
    with open(file, 'r') as f:
        reader = csv.reader(f)
        # Going through all rows
        for row in reader:
            # Adding from every row second column with name of the label
            label_list.append(row[1])
        # Deleting the first element of list because it is the name of the column
        del label_list[0]
    # Returning resulted list
    return label_list


label_list = label_text("label_names.csv")
data = None

try:
    with open('../Input/cifar10.pickle', 'rb') as file:
        data = pickle.load(file, encoding='latin1')
except IOError:
    print(colored("The dataset file you are trying to load from doesn't exist!", "red"))
    exit(0)


data.update({"labels": label_list})

pickle.dump(data, open("cifar10.pickle", "wb"))


