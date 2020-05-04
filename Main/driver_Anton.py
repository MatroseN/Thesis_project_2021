# System imports
import multiprocessing as mp
import os
# Log suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Needs to be here. Otherwise leaking occurs and bunch of annoying logs show

# 3rd party imports
from termcolor import colored

# Custom imports
import Main.modelhandler as mh

# Regulation of GPU memory usage
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Member variables
statistics = []

# Training settings
# Best width shit depth = 4 [157, 110, 76, 56] 91,51%
#Upscaled LeNet-5 by 30% = [157, 110]
# LeNet = [120, 84]
# To test with width = [[120, 84, 59, 41, 29], [157, 110, 76, 53, 37], [240, 168, 118, 82, 58], [360, 252, 177, 123, 87], [84, 59, 41, 29, 20]

models_to_train = [51, 52, 53, 54]  # What model(s) do you want to train? Leave empty to train all
width = [[3, 2, 1]]
depth = [3]
lr = []
momentum = []
batch_size = []
grid_search_models = [[50, width, depth, [0]]]
model_path = ""  # Path to model save file. If empty then training will occur
device = "GPU"  # Chose CPU or GPU to train on
verbose = 2  # 0 = Epoch results only, 1 = Regular progress bar, 2 = Epoch results and final iteration
visualize = 0  # 0 = No visualization, 1 = wrong predictions, 2 = correct predictions, 3 = all predictions
super_fast_debug_mode = False  # Gives a possibility to train faster but not as fast as dummies

# Multiprocessing the training to be able to clear GPU memory between training sessions
if __name__ == "__main__":
    mho = mh.ModelHandler("/" + device + ":0", verbose, visualize)
    models = mho.prep_models(models_to_train)
    manager = mp.Manager()
    statistics = manager.dict()  # Shared dictionary for statistics gathering
    train_single = True

    # Warn about slimmed training for debugging
    if super_fast_debug_mode:
        input(colored("WARNING!!! You are running DEBUG MODE. Confirm by pressing [Enter]. "
                      "Don't forget to focus the console window", "red"))

    for model in models:
        serial = 1
        mod = str(model).split("_", 2)
        mod = int(str(mod[1]).split(".", 2)[0])
        for i in range(len(grid_search_models)):
            if grid_search_models[i].__contains__(mod):
                for x in grid_search_models[i][1]:
                    for y in grid_search_models[i][2]:
                        for z in grid_search_models[i][3]:
                            print(colored("Training with grid search Model_" + str(grid_search_models[i][0])
                                          + ", x = " + str(x) + ", y = " + str(y) + ", z = " + str(z) +
                                          " Progress= " + str(serial) + "/"
                                          + str(len(grid_search_models[i][1]) * len(grid_search_models[i][2]
                                            * len(grid_search_models[i][3]))), 'yellow'))
                            p = mp.Process(target=mho.train_model,
                                           args=[model, model_path, statistics, x, y, z, serial, super_fast_debug_mode])
                            p.start()
                            p.join()
                            del p
                            serial += 1
                train_single = False
        if train_single:
            p = mp.Process(target=mho.train_model,
                           args=[model, model_path, statistics, 0, 0, 0, serial, super_fast_debug_mode])
            p.start()
            p.join()
            del p
        train_single = True

    mho.print_statistics(statistics)
