import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
# local modules
from common import clock, mosaic

#Parameter
SIZE = 32
CLASS_NUMBER = 3
BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 30 # num images for HOG w/BOW training
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 100 # num BOW descr for SVM training


def load_comic_dataset():
    dataset = []
    labels = []
    for comic_type in range(CLASS_NUMBER):
        comic_list = listdir("../data/TrainingImages/{}".format(comic_type))
        for comic_file in comic_list:
            if '.png' in comic_file:
                path = "../data/{}/{}".format(comic_type,comic_file)
                print(path)
                img = cv2.imread(path,0)
                img = cv2.resize(img, (SIZE, SIZE))
                img = np.reshape(img, [SIZE, SIZE])
                dataset.append(img)
                labels.append(comic_type)
    return np.array(dataset), np.array(labels)


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SIZE*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 12.5, gamma = 0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):

        return self.model.predict(samples)[1].ravel()


def evaluate_model(model, data, samples, labels):
    resp = model.predict(samples)
    print(resp)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err)*100))

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

    vis = []
    for img, flag in zip(data, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0

        vis.append(img)
    return mosaic(16, vis)

def preprocess_simple(data):
    return np.float32(data).reshape(-1, SIZE*SIZE) / 255.0


def get_hog() :
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

    return hog
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR


# TODO: Function to process HOG descriptors to BOW descriptors.
def get_bow(detector,num_clusters=40):

    # Create FLANN matcher.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Initialize BOW Kmeans trainer & BOW extractor.
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(num_clusters)
    bow_extractor = cv2.BOWImgDescriptorExtractor(detector, flann)

    return bow_kmeans_trainer,bow_extractor

def get_paths(i):
    spi_path = '../data/TrainImages/SpidermanTrainImages/spi-%d.pgm' % (i+1)
    bat_path = '../data/TrainImages/BatmanTrainImages/bat-%d.pgm' % (i+1)
    neither_path =\
            '../data/TrainImages/NeitherTrainImages/neither-%d.pgm' % (i+1)
    return spi_path, bat_path, neither_path

def add_sample(path,hog,bow_kmeans_trainer):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    descriptors = hog.compute(img, None)
    if descriptors is not None:
        bow_kmeans_trainer.add(descriptors)

def extract_bow_descriptors(img,detector,bow_extractor):
    features = detector.detect(img)
    return bow_extractor.compute(img, features)



####/ END TODO / #####


def training():

    # TODO: change to load comics images
    print('Loading data from ../data/ ... ')
    #data, labels = load_data('data.png')
    data, labels = load_comic_dataset()
    print(data.shape)
    print('Shuffle data ... ')
    # Shuffle data
    rand = np.random.RandomState(10)
    shuffle = rand.permutation(len(data))
    data, labels = data[shuffle], labels[shuffle]

    print('Deskew images ... ')
    data_deskewed = list(map(deskew, data))

    print('Defining HoG parameters ...')
    # HoG feature descriptor
    hog = get_hog()

    print('Calculating HoG descriptor for every image ... ')
    hog_descriptors = []
    for img in data_deskewed:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)



    # TODO: pass descriptors from HOG to BOW KMEANS TRAINER
    # TODO: pass vocab to BOW EXTRACTOR

    for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
        spi_path, bat_path, neither_path = get_paths(i)
        add_sample(spi_path,detector,bow_kmeans_trainer)
        add_sample(bat_path,detector,bow_kmeans_trainer)
        add_sample(neither_path,detector,bow_kmeans_trainer)

    voc = bow_kmeans_trainer.cluster()
    bow_extractor.setVocabulary(voc)

    #TODO: modify below code to use BOW descriptors....
    print('Spliting data into training (90%) and test set (10%)... ')
    train_n=int(0.9*len(hog_descriptors))
    data_train, data_test = np.split(data_deskewed, [train_n])
    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])


    print('Training SVM model ...')
    model = SVM()
    model.train(hog_descriptors_train, labels_train)

    print('Saving SVM model ...')
    model.save('data_svm.dat')
    return model

def getLabel(model, data):
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    img = [cv2.resize(gray,(SIZE,SIZE))]
    #print(np.array(img).shape)
    img_deskewed = list(map(deskew, img))
    hog = get_hog()
    hog_descriptors = np.array([hog.compute(img_deskewed[0])])
    hog_descriptors = np.reshape(hog_descriptors, [-1, hog_descriptors.shape[1]])
    return int(model.predict(hog_descriptors)[0])

