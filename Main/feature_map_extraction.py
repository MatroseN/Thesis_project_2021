import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Needs to be here. Otherwise leaking occurs and bunch of annoying logs show
from Main import modelhandler
from Main.Models import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

# Change the limit of how many predictions are needed to reach the feature image. None = All images are predicted
limit = 100

# Create an instance of model handler to get access to the evaluation data
mh = modelhandler.ModelHandler("GPU", 0, 0)

# Define what model will be used to get the feature maps and load the weights
m = models[0](mh.random_seed)
m.model.load_weights("Trained_models/Model_0_2020-03-23_15-34-03.h5")

# List the convolution layers of the loaded model
for i in range(len(m.model.layers)):
    layer = m.model.layers[i]
    if 'conv' not in layer.name:
        continue
    print("Convolution layer info: \nIndex: " + str(i),
          "\nLayer name: " + layer.name,
          "\nOutput shape: " + str(layer.output.shape))

# Set where (what layer) output will come from. This will affect the rest of the code
layer_outputs = [layer.output for layer in m.model.layers]
new_m = Model(m.model.inputs, layer_outputs)

# Predict and take out the features
feature_maps = new_m.predict(mh.data['x_test'][:limit])
feature = feature_maps[0]

# For a single print. Change the first integer in list "feature[1, :, :, i]" to pick another sign
# Chose between color or gray scale by change cmap between gray and viridis
# for i in range(len(feature)):
#     plt.imshow(feature[1, :, :, i], cmap='gray')
#     plt.show()

# Print a grid with with feature maps
rows = 2
columns = 3
ix = 1
for _ in range(int(rows)):
    for _ in range(columns):
        ax = plt.subplot(rows, columns, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(feature[44, :, :, ix-1], cmap='gray')
        ix += 1
plt.show()
