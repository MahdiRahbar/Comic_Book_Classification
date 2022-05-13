import cv2
import numpy as np
import os

if not os.path.isdir('ComicData'):
    print('ComicData folder not found.')
    exit(1)

BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 30
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 100


def get_img_paths(i):
    spi_path = 'ComicData/TrainImages/SpidermanTrainImages/spi-%d.jpg' % i
    bat_path = 'ComicData/TrainImages/BatmanTrainImages/bat-%d.png' % i
    nei_path = 'ComicData/TrainImages/NeitherTrainImages/Neither-%d.jpg' % i
    return spi_path, bat_path, nei_path

def get_test_img_paths():
    paths = []
    sub_dir_list = ["SpidermanTestImages","BatmanTestImages",\
            "NeitherTestImages"]
    for subdir in sub_dir_list:
        for filename in os.listdir('ComicData/TestImages/%s' % subdir):
            paths.append('ComicData/TestImages/'+subdir+'/'+filename)
    return paths

def add_sample(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        bow_kmeans_trainer.add(descriptors)

def extract_bow_descriptors(img):
    features = sift.detect(img)
    return bow_extractor.compute(img, features)

"""
# Create HOG descriptor extractor. <- play around w/ params to adjust accuracy
winSize = (20,20)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradient = True

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,\
        derivAperture,winSigma,histogramNormType,L2HysThreshold,\
        gammaCorrection,nlevels, signedGradient)
"""

# Create FLANN matcher.
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = {}
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Initialize BOW k-means trainer and descriptor extractor.
bow_kmeans_trainer = cv2.BOWKMeansTrainer(40) # param: num clusters

#bow_extractor = cv2.BOWImgDescriptorExtractor(hog, flann)
# (instead of this ^, need to extract features, flatten dimensionality and use
# BOW

sift = cv2.xfeatures2d.SIFT_create()
bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)

print("Training BOW extractor...")
for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
    spi_path,bat_path,nei_path = get_img_paths(i)
    add_sample(spi_path)
    add_sample(bat_path)
    add_sample(nei_path)

voc = bow_kmeans_trainer.cluster()
bow_extractor.setVocabulary(voc)

print("Training SVM classifier...")
training_data = []
training_labels = []
for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
    spi_path, bat_path, nei_path = get_img_paths(i)
    spi_img = cv2.imread(spi_path, cv2.IMREAD_GRAYSCALE)
    spi_descriptors = extract_bow_descriptors(spi_img)
    if spi_descriptors is not None:
        training_data.extend(spi_descriptors)
        training_labels.append(-1)
    bat_img = cv2.imread(bat_path, cv2.IMREAD_GRAYSCALE)
    bat_descriptors = extract_bow_descriptors(bat_img)
    if bat_descriptors is not None:
        training_data.extend(bat_descriptors)
        training_labels.append(1)
    nei_img = cv2.imread(nei_path, cv2.IMREAD_GRAYSCALE)
    nei_descriptors = extract_bow_descriptors(nei_img)
    if nei_descriptors is not None:
        training_data.extend(nei_descriptors)
        training_labels.append(0)

svm = cv2.ml.SVM_create()
svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,np.array(training_labels))

test_img_paths = get_test_img_paths()
from random import shuffle
shuffle(test_img_paths)

for test_img_path in test_img_paths[0:12]:
    img = cv2.imread(test_img_path)
    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        continue
    descriptors = extract_bow_descriptors(gray_img)
    prediction = svm.predict(descriptors)

    if prediction[1][0][0] == 1.0:
        text = 'batman'
        color = (0, 255, 0)
    else:
        text = 'not batman'
        color = (0, 0, 255)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,\
            color, 2, cv2.LINE_AA)
    cv2.imshow(test_img_path, img)

cv2.waitKey(0)
