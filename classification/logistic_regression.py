#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../tools')
from unique_math_tools import sigmoid


class LogisticRegression:
    def __init__(self):
        self.w_vector = None
        self.x_tr = None
        self.teach_labels = None
        self.num_samples = 0
        self.num_attrs = 0
        self.train_errors = list()
        self.method = 'newton'

    def phi(self, x_vectors):
        # if input is a vector
        if len(x_vectors.shape) == 1:
            return np.append(x_vectors, 1)
        else:
            return np.hstack((x_vectors, np.ones((x_vectors.shape[0], 1))))

    def predict_probability(self, x_vectors):
        feature_vectors = self.phi(x_vectors)
        return sigmoid(np.dot(feature_vectors, self.w_vector))

    def grad(self, x_vectors, teach_labels):
        error = teach_labels - self.predict_probability(x_vectors)
        w_grad = -np.mean(self.phi(x_vectors).T * error, axis=1)
        return w_grad

    def minibatch_sgd(self, eta=0.1, minibatch_size=10):
        indexes = np.arange(self.num_samples)
        np.random.shuffle(indexes)
        for index in np.arange(0, self.num_samples, minibatch_size):
            minibatch_indexes = indexes[index: index + minibatch_size]
            x_minibatch = self.x_tr[minibatch_indexes]
            t_minibatch = self.teach_labels[minibatch_indexes]
            w_grad = self.grad(x_minibatch, t_minibatch)
            w_vector_new = self.w_vector - eta * w_grad
            self.train_errors.append(np.mean(np.abs(self.teach_labels - self.predict_probability(self.x_tr))))
            diff = np.linalg.norm(w_vector_new - self.w_vector) / np.linalg.norm(self.w_vector)
            if diff < 0.001:
                break
            self.w_vector = w_vector_new
        eta *= 0.9

    def newton_method(self):
        predicted_probabilities = self.predict_probability(self.x_tr)
        self.train_errors.append(np.mean(np.abs(self.teach_labels - predicted_probabilities)))
        R = np.diag(predicted_probabilities * (1 - predicted_probabilities))
        phi_vectors = self.phi(self.x_tr)
        #calculate Hessian
        H = np.dot(phi_vectors.T, np.dot(R, phi_vectors))
        w_vector_new = self.w_vector - np.dot(np.linalg.inv(H), np.dot(phi_vectors.T, (predicted_probabilities - self.teach_labels)))
        diff = np.linalg.norm(w_vector_new - self.w_vector) / np.linalg.norm(self.w_vector)
        self.w_vector = w_vector_new
        return diff

    def estimate_weight(self, n=100):
        if self.method == 'newton':
            for _ in np.arange(n):
                diff = self.newton_method()
                if diff < 0.001:
                    break
        else:
            for _ in np.arange(n):
                self.minibatch_sgd()

    def fit(self, x_tr, teach_labels, method='newton'):
        self.x_tr = x_tr
        self.num_samples, self.num_attrs = self.x_tr.shape
        self.w_vector = np.random.rand(self.num_attrs + 1)
        self.teach_labels = teach_labels
        self.method = method
        self.estimate_weight()


if __name__ == '__main__':
    num_of_sample = 1000
    num_of_attributes = 4
    np.random.seed(0)
    X = np.random.randn(num_of_sample, num_of_attributes)

    def f(x):
        return 5 * x[0] + 3 * x[1] + 4 * x[2] + x[3] - 1
    T = np.array([1 if f(x) > 0 else 0 for x in X])

    model = LogisticRegression()
    model.fit(X, T)
    print(model.w_vector)





