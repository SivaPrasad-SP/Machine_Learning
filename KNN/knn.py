import numpy as np
from collections import Counter

class Knn:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    @staticmethod
    def euclidean_dist(p, q):
        # euclidean distance between 2 points of any dimension
        summation = np.sum((p-q)**2)
        return np.sqrt(summation)
    
    def predict(self, x_test):
        # calc distance betwen every test point and all train points
        labels = []
        for xt in x_test:
            distances = [self.__class__.euclidean_dist(xt, xtr) for xtr in self.x_train]
            top_k_idx = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[idx] for idx in top_k_idx]
            most_common = Counter(k_labels).most_common(1) # return list of tuples
            labels.append(most_common[0][0]) # most_common[first_tuple][first_element]
        return labels
    
    def accuracy(self, y_test, y_pred):
        accuracy = np.sum(y_test == y_pred)/len(y_pred)
        return round(accuracy*100, 2)