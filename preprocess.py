import os 
import sys
import numpy as np 
import cv2

from skimage.feature import hog

import matplotlib.pyplot as plt


HEIGHT = 256
WIDTH = 128

def get_file_names(path):
    """ 
    collect all the images path. 
    """
    return [os.path.join(path,f) for f in os.listdir(path)]


def pack_data(data_dir, class_list, label_dict, shuffle = True):
    """
    Find all training image pathes, and return images, with their
        corresponding labels. 
    """
    image_paths = {}
    for training_name in class_list:
        dir_= os.path.join(data_dir, training_name)
        class_path = get_file_names(dir_)
        image_paths[training_name] = class_path
    index_to_class = {}
    index_to_class = {v:k for k,v in label_dict.items()}
    image_data = []
    for k, v in image_paths.items():
        for im in v: 
            image_data.append((im, label_dict[k]))
    if shuffle == True: 
        np.random.shuffle(image_data)
    image_paths, labels = zip(*image_data)
    return image_paths, labels, index_to_class


def get_desc(image_paths, width = WIDTH, height=  HEIGHT ):
    """
    get images Hog descriptor features. 
    """
    des_list=[]
    for path in image_paths:
        img = cv2.imread(path)
        resized_image = []
        for i in range(3): 
            resized_image.append(cv2.resize(img[:,:,i], (width, height)))
        resized_image = np.transpose(np.array(resized_image), (1, 2, 0))
        # resized_image = cv2.GaussianBlur(resized_image, (3, 3), 7)
        descriptor, hog_image = hog(resized_image, orientations=9, pixels_per_cell=(4, 4), 
                        cells_per_block=(2, 2), visualize=True,channel_axis = -1,feature_vector= False)
        des_list.append((path,np.array(descriptor)))

    return des_list


def get_BoW(des_list, label_dict):
    """
    Collect BoW.
    """
    reshaped_features = []
    image_labels = []
    BoW = []
    BoW_labels = [] 
    for i in range(len(des_list)): 
        temp_des = des_list[i][1]
        reshaped_features.append(temp_des.reshape(int((HEIGHT/4-1)*(WIDTH/4-1)),-1))
        image_labels.append(label_dict[des_list[i][0].split('/')[2]])
        for j in range(reshaped_features[i].shape[0]):
            BoW.append(reshaped_features[i][j])
            BoW_labels.append(des_list[i][0].split('/')[2])

    return BoW, BoW_labels, reshaped_features, image_labels 

def build_histogram(descriptor_list, cluster_alg):
    """
    Calculates the histogram of the featuers. 
    """
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram


def make_features(feature_desc, clustering_model):  
    """
    create BoW features from hog features using a
        clustering algorithm such as kmeans. 
    # reshaped_features
    """
    preprocessed_image = []
    for descriptor in feature_desc:
        if (descriptor is not None):
            temp_features = []
            for piece in descriptor: 
                # getting histogram for each piece 
                histogram = build_histogram(piece.reshape(1,-1), clustering_model)  
                temp_features.append(histogram)
            temp_features = np.sum(np.array(temp_features), 0)
            preprocessed_image.append(temp_features.reshape(-1,))

    return preprocessed_image


def matching_points(train_image, query_image):
    """
    Finding the matching points in the test image with comic books.
    """
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(train_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(query_image, None)
    FLAN_INDEX_KDTREE = 0
    index_params = dict (algorithm = 0, trees=5) # algorithm = FLAN_INDEX_KDTREE
    search_params = dict (checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch (descriptors1, descriptors2, k=2)
    good_matches = []
    distance_list = []
    for m, n in matches:
        distance_list.append(m.distance)
        if m.distance < 0.65* n.distance:
            good_matches.append([m])
    average_distance = np.mean(distance_list)

    good_matches = sorted(good_matches, key = lambda x:x[0].distance)
    good_matches = good_matches[:50]
    query_pts = np.float32([keypoints1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    train_pts = np.float32([keypoints2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return query_pts, train_pts, average_distance

def create_mosaic_range():
    """
    This function creates ranges for making the mosaic. 
    """
    range_dictionary = {}
    maximum_height = 1280
    maximum_width = 960

    h = 0 
    w = 0
    slide_w = 50
    counter = 0 
    window_dim = 400 
    reduce_width = 50
    add_height = 100

    h_end = window_dim + add_height
    w_end = window_dim - reduce_width

    while h_end< maximum_height: 

        new_h = h  + window_dim + add_height
        w = 0 
        w_end = window_dim
        while w_end<maximum_width: 
            new_w = w + window_dim - reduce_width
            range_dictionary[counter] = ((h, new_h),(w, new_w))
            counter += 1
            w += slide_w
            w_end += slide_w 
        
        h += slide_w 
        h_end += slide_w  

    return range_dictionary


def mosaic(range_dictionary,train_image, query_image):
    """
    This function is to make mosaic on the image and 
        basicaly devide it into smaller pieces. 
    """
    min_dist = np.inf 
    for p_key in range_dictionary.keys():
        i1,i2 = range_dictionary[p_key][0]
        j1,j2 = range_dictionary[p_key][1]
        query_pts, train_pts, average_dist = matching_points(train_image, query_image[i1:i2, j1:j2])
        if average_dist < min_dist: 
            min_dist = average_dist
            best_pts = train_pts 
            best_key = p_key 
    return best_pts, best_key, min_dist


def plot(range_dictionary, best_key, test_image):
    """
    This function is to plot the block of the test image
        which the comic book is located.
    """
    i1,i2 = range_dictionary[best_key][0]
    j1,j2 = range_dictionary[best_key][1]
    test_image[i1:i2, j1:j2]
    target_image = test_image[i1:i2, j1:j2, ::-1]

    plt.figure(figsize=(15,9))  # 20,12
    plt.imshow(target_image)
    plt.show() 

    return target_image


def preprocess_test_image(target_image, clustering_model, path = ''):
    """
    Preprocess the test image all at once and return the final features.
    """
    resized_image = []
    des_list = []
    for i in range(3): 
        resized_image.append(cv2.resize(target_image[:,:,i], (WIDTH, HEIGHT)))
    resized_image = np.transpose(np.array(resized_image), (1, 2, 0))
    # resized_image = cv2.GaussianBlur(resized_image, (3, 3), 7)
    descriptor, hog_image = hog(resized_image, orientations=9, pixels_per_cell=(4, 4), 
                    cells_per_block=(2, 2), visualize=True,channel_axis = -1,feature_vector= False)
    #     descriptor = hog.compute(resized_image)
    des_list.append((path,np.array(descriptor)))
    reshaped_features = []

    for i in range(len(des_list)): 
        temp_des = des_list[i][1]
        reshaped_features.append(temp_des.reshape(int((HEIGHT/4-1)*(WIDTH/4-1)),-1))

    preprocessed_image = []
    for descriptor in reshaped_features:
        if (descriptor is not None):
            temp_features = []
            for piece in descriptor: 
                histogram = build_histogram(piece.reshape(1,-1), clustering_model)  # getting histogram for each piece 
                temp_features.append(histogram)
            temp_features = np.sum(np.array(temp_features), 0)
            preprocessed_image.append(temp_features.reshape(-1,))
    return preprocessed_image[0].reshape(1, -1) 