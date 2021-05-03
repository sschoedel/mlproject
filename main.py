import numpy as np
import os
from naive_bayes import train_naive_bayes, test_naive_bayes
from cnn.tester import test_cnn
from mlp.tester import Test_MLP
from sklearn.metrics import classification_report
import sys, getopt
from cnn.res_cnn import ResNet

# path constants
classes = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Crosswalk', 'Hydrant', 'Palm', 'Traffic Light']
TRAIN_PATH = './Train/'
TEST_PATH = './Test/'

def main(argv):

    # option flags
    train_cnn_flag = False
    train_mlp_flag = False

    # read in command args
    try:
        opts, args = getopt.getopt(argv, "hc")
    except getopt.GetoptError:
      print("main.py -h for help")
      sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("-h\tHelp")
            print("-c\tTrain CNN (takes a couple hours)")
            sys.exit()
        elif opt == "-c":
            train_cnn_flag = True
        elif opt == "-m":
            train_mlp_flag = True

    test_images_truth = []
    # traverse the test image directories to get the ground truth
    dir_i = 0
    for dir_name in classes:  
        for file_name in os.listdir(TEST_PATH + dir_name):
            test_images_truth.append(dir_i)
        dir_i += 1
    test_images_truth = np.array(test_images_truth)

    for i in range(10):
        num = i + 2

        print("======= Running Naive Bayes ========")
        print("-- Begin training NB --")
        print(num)
        naive_bayes_model = train_naive_bayes(classes, num)

        predictions = []
        print("-- Begin testing NB --")
        dir_i = 0
        for dir_name in classes:  
            for file_name in os.listdir(TEST_PATH + dir_name):
                img_path = TEST_PATH + dir_name + "/" + file_name
                res = test_naive_bayes(naive_bayes_model, classes, img_path, dir_i, num)
                predictions.append(res)
            dir_i += 1
        predictions = np.array(predictions)

        # Output results of Naive Bayes
        print("Naive Bayes Report:")
        print(classification_report(test_images_truth, predictions, digits=3))

    '''
    print("======= Running CNN ========")


    #print("-- begin training CNN --")
    #naive_bayes_model = train_naive_bayes(classes)

   '''


    print("-- Begin testing CNN--")
    labels, predictions = test_cnn()
    
    # Output results of Naive Bayes
    print("CNN Report:")
    print(classification_report(labels, predictions, digits=3))


    mlp = Test_MLP()
    
    if train_mlp_flag:
        print("-- begin training MLP --")
        mlp.train_mlp()
    
    print(" -- begin testing MLP --")
    test_image_predictions = mlp.test_mlp()

if __name__ == "__main__":
    main(sys.argv[1:])
