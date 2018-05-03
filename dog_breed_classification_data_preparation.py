# -*- coding: utf-8 -*-
"""
minhthien nguyen
data preparation script for dog classification project
"""

""" 1 """
""" Since breed is categorical  data we need to one hot encode this """
import pandas as pd
import numpy as np
dirPath = "C:/Users/thien nguyen/Desktop/dog_breed_data/"
labelFile = "labels.csv"
labelData = pd.read_csv(dirPath + labelFile)
print(labelData.head(5))

""" 2 """
""" one_hot_encode now have the dog breed as column and  
the value of 1 in the column mean the row are classied at the breed. 
one_hot_labels is in an array form of one_hot_encode """
one_hot_encode  = pd.get_dummies(labelData['breed'])
print(one_hot_encode.head(1))
one_hot_labels = np.asarray(one_hot_encode)
print(one_hot_labels[1:2,])
np.save('one_hot_encode_label.npy', one_hot_encode.columns.values)


""" 3 """
""" Try to read an image in and see if we can read it """
img_rows=128
img_cols=128
num_channel=3    

import cv2
import matplotlib.pyplot as plt 

train_img_dir = dirPath + 'train/'
img_1 = cv2.imread(train_img_dir + '0a0c223352985ec154fd604d7ddceabd.jpg')
plt.title('Original Image')
plt.imshow(img_1)

""" 4 """
""" we resize an image to a smaller dimension we will eventually want to do this 
for all images so we get the consitent size of all images """
img_1_resize= cv2.resize(img_1, (img_rows, img_cols)) 
print (img_1_resize.shape)
plt.title('Resized Image')
plt.imshow(img_1_resize)

""" 5 """
""" Create the full path for all trainning images """
from os import listdir
from os.path import join

image_paths = list()
for f in listdir(train_img_dir):
   image_paths.append(join(train_img_dir,f))

print(image_paths[:10])
print(len(image_paths))


""" 6 """
""" Resize and read all training images into x_feautre and the coresspond one_hot_labels into y_feature . 
as we can see the trainning images is sotred ad numpy array """
x_feature = []
y_feature = []

i = 0
for img in image_paths:
    train_img = cv2.imread(img)
    label = one_hot_labels[i]
    train_img_resize = cv2.resize(train_img, (img_rows, img_cols)) 
    x_feature.append(train_img_resize)
    y_feature.append(label)
    i += 1
    
print(x_feature[1])   

""" 7 """
""" normalize the RGB values """
x_train_data = np.array(x_feature)
x_train_data = x_train_data.reshape(x_train_data.shape[0], img_rows, img_cols, 3).astype( 'float32' )
x_train_data = x_train_data / 255
print(x_train_data.shape)
print(x_train_data[1])

y_train_data = np.array(y_feature)
print (y_train_data.shape)
print (y_train_data[1])


""" 8 """
""" split train data to train and validtaion 70% train and 30% validation """
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train_data, y_train_data, test_size=0.3, random_state=2)

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(x_val.shape)

""" 9 """
""" Get all the test images full path and the unique id """
#prepare test data
test_img_dir = dirPath + 'test/'
test_image_paths = list()
test_image_ids = list()

for f in listdir(test_img_dir):
   test_image_paths.append(join(test_img_dir,f))
   test_image_ids.append(f.rsplit('.')[0])
   
print(test_image_ids[1:3])   
np.save('test_image_id.npy', test_image_ids)   

print(test_image_paths[:3])
print(len(test_image_paths))

""" 10 """
""" Read and normalize test images into x_test_feature """
x_test_feature = []

for img in test_image_paths: # f for format ,jpg
    test_img = cv2.imread(img)
    test_img_resize = cv2.resize(test_img, (img_rows, img_cols)) 
    x_test_feature.append(test_img_resize)
    
x_test_data = np.array(x_test_feature)
x_test_data = x_test_data.reshape(x_test_data.shape[0], img_rows, img_cols, 3).astype( 'float32' )
x_test_data = x_test_data / 255
print(x_test_data.shape)    

""" 11 """
""" we need to save all these data to loaded in colab to use the gpu """
np.save('xtrain.npy', x_train)
np.save('ytrain.npy', y_train)
np.save('xval.npy', x_val)
np.save('yval.npy', y_val)
np.save('xtestdata.npy', x_test_data)

















