
import sys
import os
import pickle


from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score



def train_clustering(input_data, clustering_calgo = KMeans , K = 256):
    """
    input_data: The data which we apply clustering on.
    clustering_algo: The algorithm for clustering
    K : number of clusters 
    """
    model_name = 'kmeans_model.pkl'
    if os.path.exists(model_name): 
        model = pickle.load(open(model_name, 'rb'))
        return model 
    else: 
        model = clustering_calgo(n_clusters = K)  # %56 with K=128
        model.fit(input_data)
        pickle.dump(model, open(model_name, 'wb'))
        return model


def SVM(preprocessed_image, image_labels):
    """
    Load SVM if a trained model file was available, 
        otherwise, train a new one using Grid Search 
        with Cross Validation of 5 folds to find the 
        most accurate hyperparameter for the 
        classification.
    """

    x_train, x_test, y_train, y_test = train_test_split(preprocessed_image, image_labels,
                                                test_size = .2, random_state = 14)
    grid = { 
    'C': [2**(-5), 2**(-4), 2**(-3), 2**(-2), 2**(-1), 1, 2, 4 ,8 ,16],
    'kernel': ['linear', 'poly', 'rbf'],
    }
    pickle_file_name = 'best_param.pkl'
    model_name = 'model.sav'
    if os.path.exists(model_name): 
        svm_clf = pickle.load(open(model_name, 'rb'))
        y_pred = svm_clf.predict(x_test)
        print("The test accuracy of the SVM model is: %", accuracy_score(y_test, y_pred)*100)
        return svm_clf
    if os.path.exists(pickle_file_name):
        with open(pickle_file_name, 'rb') as handle:
            best_param = pickle.load(handle)
            svm_clf = SVC(**best_param)
            svm_clf.fit(x_train, y_train)
            y_pred = svm_clf.predict(x_test)
            print("The test accuracy of the SVM model is: %",accuracy_score(y_test, y_pred)*100)
            pickle.dump(svm_clf, open(model_name, 'wb'))
            return svm_clf
    else: 
        svm_cv = GridSearchCV(estimator=SVC(), param_grid=grid, cv= 5)
        svm_cv.fit(x_train, y_train)
        best_param = svm_cv.best_params_
        with open(pickle_file_name, 'wb') as handle:
            pickle.dump(svm_cv.best_params_, handle, protocol=pickle.HIGHEST_PROTOCOL)
        svm_clf = SVC(**best_param)
        svm_clf.fit(x_train, y_train)
        y_pred = svm_clf.predict(x_test)
        print("The test accuracy of the SVM model is: %",accuracy_score(y_test, y_pred)*100)
        pickle.dump(svm_clf, open(model_name, 'wb'))
        return svm_clf