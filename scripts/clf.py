import numpy as np
import scipy
import sklearn
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def print_pairs(a, b):
    for x, y in zip(a, b):
        print(x, y)


class SVM(object):
    def __init__(self, kernel='poly'):
        self.clf = SVC(kernel=kernel)

    def train(self, data, labels):
        self.clf.fit(data, labels)

    def test(self, data, labels):
        return self.clf.predict(data), labels


# Doesn't work
class SVM_chi2(object):
    def __init__(self):
        self.clf = SVC(kernel='precomputed')
        # self.clf = SVC(kernel=chi2_kernel)

    def train(self, data, labels):
        self.train_data = data
        K = chi2_kernel(data, gamma=.000001)
        self.clf.fit(K, labels)

    def test(self, data, labels):
        K = chi2_kernel(data, self.train_data, gamma=.000001)
        pred = self.clf.predict(K)
        print_pairs(pred, labels)
        return pred, labels


# K nearest neighbors
class KNN(object):
    def __init__(self, n=9):
        self.clf = sklearn.neighbors.NearestNeighbors(n_neighbors=n, metric='braycurtis')
        # self.clf = sklearn.neighbors.NearestNeighbors(n_neighbors=n, metric='canberra')

    def train(self, data, labels):
        self.clf.fit(data, labels)
        self.labels = labels

    def test(self, data, labels):
        res = [0] * len(labels)
        for i, vect in enumerate(data):
            closest = self.clf.kneighbors(vect.reshape([1, -1]), return_distance=False)
            items, count = scipy.stats.mode([self.labels[l] for l in closest])
            res[i] = items[0][max(range(len(items)), key=lambda x: count[x])]
            # res[i] = max(set(closest_labels), key=closest_labels.count)

        return res, labels


# Fully connected neural network classifier
class MLP(object):
    def __init__(self, hidden_layers=(64, 16)):
        self.clf = MLPClassifier(hidden_layer_sizes=hidden_layers, solver='lbfgs', activation='relu',
                                 learning_rate='constant', verbose=False)

    def train(self, data, labels):
        self.clf.fit(data, labels)

    def test(self, data, labels):
        return self.clf.predict(data), labels
