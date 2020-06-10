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
models_to_train = [0, 11, 12]  # What model(s) do you want to train? Leave empty to train all
width = []
depth = []
lr = [0.001, 0.005, 0.01, 0.015, 0.02]
momentum = [0, 0.7, 0.8, 0.9, 0.95, 0.99]
batch_size = [8, 16, 32, 64, 128]
grid_search_models = [[11, lr, momentum, [0]], [12, [0], [0], batch_size]]  # [[10, lr, momentum, batch_size]]
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

    seed_serial = -1
    if super_fast_debug_mode:
        mho.data = mho.data[:2]
    for seed in mho.seeds:
        seed_serial += 1
        for model in models:
            serial = 1
            mod = str(model).split("_", 2)
            mod = int(str(mod[1]).split(".", 2)[0])
            for i in range(len(grid_search_models)):
                if grid_search_models[i].__contains__(mod):
                    for x in grid_search_models[i][1]:
                        for y in grid_search_models[i][2]:
                            for z in grid_search_models[i][3]:
                                print(colored("Seed index: " + str(seed_serial) + " Seed:" + str(seed), 'yellow'))
                                print(colored("Training with grid search Model_" + str(grid_search_models[i][0])
                                              + ", x = " + str(x) + ", y = " + str(y) + ", z = " + str(z) +
                                              " Progress= " + str(serial) + "/"
                                              + str(len(grid_search_models[i][1]) * len(grid_search_models[i][2]
                                                * len(grid_search_models[i][3]))), 'yellow'))
                                if super_fast_debug_mode and serial > 2:
                                    serial += 1
                                    print(colored("Skipping due to debug mode", 'red'))
                                    pass
                                else:
                                    p = mp.Process(target=mho.train_model,
                                                   args=(model, model_path, statistics, x, y, z, serial, seed_serial, seed,
                                                         super_fast_debug_mode))
                                    p.start()
                                    p.join()
                                    del p
                                    serial += 1
                    train_single = False
            if train_single:
                print(colored("Seed index: " + str(seed_serial) + " Seed:" + str(seed), 'yellow'))
                print(colored("Training Model_" + str(mod), 'yellow'))
                p = mp.Process(target=mho.train_model,
                               args=(model, model_path, statistics, 0, 0, 0, serial, seed_serial, seed,
                                     super_fast_debug_mode))
                p.start()
                p.join()
                del p
            train_single = True

    mho.print_statistics(statistics)
