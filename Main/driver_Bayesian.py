# System imports
import multiprocessing as mp
import os

# Log suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Needs to be here. Otherwise leaking occurs and bunch of annoying logs show

# 3rd party imports
from termcolor import colored
import matplotlib.pyplot as plt
from skopt import space, gp_minimize, plots

# Custom imports
import Main.modelhandler as mh

# Regulation of GPU memory usage
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Member variables
statistics = []

# Training settings
# Best width shit depth = 4 [157, 110, 76, 56] 91,51%
# Upscaled LeNet-5 by 30% = [157, 110]
# LeNet = [120, 84]
# To test with width = [[[120, 84, 59, 41, 29], [84, 59, 41, 29, 20], [157, 110, 76, 53, 37], [240, 168, 118, 82, 58], [360, 252, 177, 123, 87]]

lr_low = 1e-3
lr_upp = 2e-2
momentum_low = 0
momentum_upp = 99e-2

default_parameters = [0.01, 0.9]
lr = space.Real(low=lr_low, high=lr_upp, prior='log-uniform', name='lr')
momentum = space.Real(low=momentum_low, high=momentum_upp, prior='log-uniform', name='momentum')
dimensions = [lr, momentum]

batch_size = []
model_path = ""  # Path to model save file. If empty then training will occur
device = "CPU"  # Chose CPU or GPU to train on
verbose = 2  # 0 = Epoch results only, 1 = Regular progress bar, 2 = Epoch results and final iteration
visualize = 0  # 0 = No visualization, 1 = wrong predictions, 2 = correct predictions, 3 = all predictions
super_fast_debug_mode = False  # Gives a possibility to train faster but not as fast as dummies

# Multiprocessing the training to be able to clear GPU memory between training sessions
if __name__ == "__main__":
    mho = mh.ModelHandler("/" + device + ":0", verbose, visualize)
    manager = mp.Manager()
    statistics = manager.dict()  # Shared dictionary for statistics gathering
    train_single = True

    # Warn about slimmed training for debugging
    if super_fast_debug_mode:
        input(colored("WARNING!!! You are running DEBUG MODE. Confirm by pressing [Enter]. "
                      "Don't forget to focus the console window", "red"))
    if super_fast_debug_mode:
        mho.data = mho.data[:2]

    print(colored("Training Model_70", 'yellow'))

    search_results = mho.bayesian_optimize(_STOCK="DJI", _INTERVALL="", _TYPE="Daily")

    # Best parameters found
    search_results.x
    print("Best Parameters found")
    print(search_results.x)
    print()

    plots.plot_convergence(search_results)
    plt.savefig("Converge.png", dpi=800)
