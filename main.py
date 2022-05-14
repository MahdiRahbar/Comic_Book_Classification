#!/usr/bin/python

# import os 
# import sys 
# import math
import numpy as np 
# import random 
import argparse
import warnings
import cv2
# import matplotlib.pyplot as plt
# import pickle
# import pylab as pl

# from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.svm import SVC
# from skimage.feature import hog
# from skimage import exposure

from preprocess import *
from train import * 
from predict import * 

np.random.seed(42)



parser = argparse.ArgumentParser(description='A program to find comic books and classify the heros! Happy Coding! Yayyyy!!!! :D')

parser.add_argument("-p", "--path", help="Path to the .", default= './data/test_comic_locations/spiderman1.jpeg')
args = parser.parse_args()

data_dir = './data/'
class_list = ['Batman', 'Spiderman', 'None']
label_dict = {'Batman':0, 'Spiderman':1, 'None':2}

warnings.filterwarnings("ignore")

HEIGHT = 256
WIDTH = 128
CLUSTER_NUMBER = 256


test_path = args.path
training_path = './data/training.png'

if __name__ == "__main__":
    image_paths, labels, index_to_class = pack_data(data_dir, class_list, label_dict, shuffle = True)
    des_list = get_desc(image_paths, width = WIDTH, height =  HEIGHT )
    BoW, BoW_labels, reshaped_features, image_labels = get_BoW(des_list, label_dict)
    kmeans_model = train_clustering(np.array(BoW), clustering_calgo = KMeans , K = CLUSTER_NUMBER)
    preprocessed_image = make_features(feature_desc=reshaped_features, clustering_model = kmeans_model)

    svm_clf = SVM(preprocessed_image, image_labels)

    query_image = cv2.imread(test_path)
    train_image = cv2.imread(training_path)
    query_pts, train_pts, average_dist = matching_points(train_image, query_image)

    range_dictionary = create_mosaic_range()
    best_pts, best_key, min_dist = mosaic(range_dictionary, train_image, query_image)
    target_image = plot(range_dictionary, best_key, query_image)

    preprocessed_data = preprocess_test_image(target_image, kmeans_model, path = '')
    y_pred = svm_clf.predict(preprocessed_data)

    print("The predicted image class is: ", index_to_class[int(y_pred)])
    print("Hope it predicted the right class! :)")