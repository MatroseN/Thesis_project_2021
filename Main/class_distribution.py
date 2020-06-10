import Main.modelhandler as model_handler

'''
This file is used to check class distribution
'''

mh = model_handler.ModelHandler("Dummy", 0, 0)

dist_train = [0] * 43
dist_valid = [0] * 43
dist_test = [0] * 43
dist_total = [0] * 43

# Populate the lists with number of occurrences
for i in range(len(mh.data['x_train'])):
    dist_train[mh.data['y_train'][i]] += 1
    dist_total[mh.data['y_train'][i]] += 1

for j in range(len(mh.data['x_validation'])):
    dist_valid[mh.data['y_validation'][j]] += 1
    dist_total[mh.data['y_validation'][j]] += 1

for k in range(len(mh.data['x_test'])):
    dist_test[mh.data['y_test'][k]] += 1
    dist_total[mh.data['y_test'][k]] += 1

# Print the collected data
print("\n\nLists above each other for easier comparison")
print("Train\t\t" + str(dist_train))
print("Validation\t" + str(dist_valid))
print("Test\t\t" + str(dist_test))
print("Total\t\t" + str(dist_total))

# Sum up the different partitions
samples_train = 0
samples_validation = 0
samples_test = 0
samples_total = 0

for x in dist_train:
    samples_train += x

for y in dist_valid:
    samples_validation += y

for z in dist_test:
    samples_test += z

for t in dist_total:
    samples_total += t

print("\Train\tValidation\tTest\tTotal")
print(str(samples_train) + "\t" + str(samples_validation) + "\t\t" + str(samples_test) + "\t" + str(samples_total))