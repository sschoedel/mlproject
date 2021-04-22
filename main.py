import numpy as np
import os
from naive_bayes import train_naive_bayes, test_naive_bayes
from sklearn.metrics import classification_report


classes = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Crosswalk', 'Hydrant', 'Palm', 'Traffic Light']
TRAIN_PATH = './Train/'
TEST_PATH = './Test/'

test_images_truth = []
# traverse the test image directories to get the ground truth
dir_i = 0
for dir_name in classes:  
    for file_name in os.listdir(TEST_PATH + dir_name):
        test_images_truth.append(dir_i)
    dir_i += 1
test_images_truth = np.array(test_images_truth)

print("======= Running Naive Bayes ========")
# initalize metrics
true_positives_nb = 0
false_positives_nb = 0
true_negatives_nb = 0
false_negatives_nb = 0

print("-- begin training NB --")
naive_bayes_model = train_naive_bayes(classes)

predictions = []
print("-- begin testing NB --")
dir_i = 0
for dir_name in classes:  
    for file_name in os.listdir(TEST_PATH + dir_name):
        img_path = TEST_PATH + dir_name + "/" + file_name
        res = test_naive_bayes(naive_bayes_model, classes, img_path, dir_i)
        predictions.append(res)
    dir_i += 1
predictions = np.array(predictions)

# Output results of Naive Bayes
print("Naive Bayes Report:")
print(classification_report(test_images_truth, predictions, digits=3))


