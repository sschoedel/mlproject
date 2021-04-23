import numpy as np
import h5py
import os
import cv2

dir_names = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Crosswalk', 'Hydrant', 'Palm', 'Traffic Light']
train_path = './Train/'
test_path = './Test/'

classes_enum = {
    'Bicycle':0,
    'Bridge':1,
    'Bus':2,
    'Car':3,
    'Crosswalk':4,
    'Hydrant':5,
    'Palm':6,
    'Traffic Light':7
}

# Create fxn dataset with f=120*120=14,400 features and n=num_images examples

all_train_images = np.array([[cv2.imread(train_path + dir_name + '/' + img_name) for img_name in os.listdir(train_path + dir_name)] for dir_name in dir_names], dtype=object)
all_test_images = np.array([[cv2.imread(test_path + dir_name + '/' + img_name) for img_name in os.listdir(test_path + dir_name)] for dir_name in dir_names], dtype=object)

train_data = np.vstack([train_img.flatten() for i,dir_name in enumerate(dir_names) for train_img in all_train_images[i] if train_img.shape[0] == 120])
train_data = train_data.astype('float64')
train_data = np.transpose(train_data)
print(train_data)
print(train_data.shape)
test_data = np.vstack([test_img.flatten() for i,dir_name in enumerate(dir_names) for test_img in all_test_images[i] if test_img.shape[0] == 120])
test_data = test_data.astype('float64')
test_data = np.transpose(test_data)
print(test_data)
print(test_data.shape)

train_labels = np.array([classes_enum[dir_name] for i,dir_name in enumerate(dir_names) for train_img in all_train_images[i] if train_img.shape[0] == 120])
print(train_labels)
print(train_labels.shape)
test_labels = np.array([classes_enum[dir_name] for i,dir_name in enumerate(dir_names) for test_img in all_test_images[i] if test_img.shape[0] == 120])
print(test_labels)
print(test_labels.shape)

data_file = h5py.File('recaptcha_data.h5', 'w')

data_file.create_dataset('train_data', data=train_data)
data_file.create_dataset('test_data', data=test_data)
data_file.create_dataset('train_labels', data=train_labels)
data_file.create_dataset('test_labels', data=test_labels)

data_file.close()