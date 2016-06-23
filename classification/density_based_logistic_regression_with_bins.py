#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from logistic_regression import LogisticRegression


class DensityBasedLogisticRegressionWithBins(LogisticRegression):
    def __init__(self):
        LogisticRegression.__init__(self)
        # 各特徴が数値データなら0,カテゴリカルデータなら1を要素に持つ配列
        self.x_types = None
        # 数値データ,カテゴリカルデータそれぞれの中での番号を要素に持つ配列
        self.x_types_index = None
        self.band_width_vector = None
        self.numerical_index = None
        self.categorical_index = None
        self.feature_vectors_for_numeric = None
        self.feature_vectors_for_category = None
        self.max_values = None
        self.min_values = None
        self.num_of_bins = 0
        self.bin_length = None

    def set_x_types_index(self):
        self.x_types_index = np.empty(self.dim_features)
        # 数値データ内での特徴の番号
        self.numerical_index = np.where(self.x_types == 0)[0]
        # カテゴリカルデータ内での特徴の番号
        self.categorical_index = np.where(self.x_types == 1)[0]
        self.num_of_numerical_features = len(self.numerical_index)
        self.num_of_categorical_features = len(self.categorical_index)
        # 数値データ,カテゴリカルデータそれぞれの中での番号を保持
        self.x_types_index[self.numerical_index] = np.arange(self.num_of_numerical_features)
        self.x_types_index[self.categorical_index] = np.arange(self.num_of_categorical_features)
        self.x_types_index = self.x_types_index.astype('int64')

    def initialize_bandwidth_vector(self):
        # 今のところcategoricalにもnumericalにも定義している
        # h_d = (1.06 * sigma(d) * N)^(-1/5)
        sigma = np.std(self.x_tr, ddof=1, axis=0)
        self.band_width_vector = np.power(1.06 * sigma * self.num_samples, (-1 / 5))

    def initialize_num_of_bins(self):
        # ビンの数は100で固定
        self.num_of_bins = 100

    def create_bin_model(self):
        # 数値データの列から、最大値・最小値を取得しビン幅を算出
        self.max_value = np.amax(self.x_tr[:, self.numerical_index], axis=0)
        self.min_value = np.amin(self.x_tr[:, self.numerical_index], axis=0)
        self.bin_length = (self.max_value - self.min_value) / self.num_of_bins
        # それぞれの数値データに対応するビンインデックスを取得
        bin_index_for_numerical = self.get_bin_index_for_numerical(self.x_tr[:, self.numerical_index])
        # 数値データ、カテゴリカルデータそれぞれの特徴ベクトルを保持するための配列・リストを用意
        self.feature_vectors_for_numeric = np.empty((self.num_of_bins, self.num_of_numerical_features))
        self.feature_vectors_for_category = [0] * self.num_of_categorical_features
        # ビンカーネルを計算するための100*100の配列を作成
        tmp = np.tile(np.arange(100), 100).reshape((100, 100))
        tmp_dist = np.abs(tmp.T - tmp)
        # 現段階ではfeature_vectors_for_numeric は shape(ビン数,特徴数) でcategoryが(特徴数,ビン数)
        # numericの方はfor文使わなくても書けるはず
        for d in np.arange(self.dim_features):
            if self.x_types[d] == 0:
                attr_index = self.x_types_index[d]
                # ヒストグラムの作成
                hist0, _ = np.histogram(bin_index_for_numerical[:, attr_index][np.where(self.teach_labels == 0)],
                                        np.arange(0, self.num_of_bins + 1))
                hist1, _ = np.histogram(bin_index_for_numerical[:, attr_index][np.where(self.teach_labels == 1)],
                                        np.arange(0, self.num_of_bins + 1))
                # bin_lengthやband_widthをnumericalだけに定義するなら変更の必要あり
                # ビンカーネルの計算
                bin_kernel = np.exp(
                    - np.power(tmp_dist * self.bin_length[attr_index] / self.band_width_vector[d], 2) / 2)
                # 特徴量(phi)の計算
                self.feature_vectors_for_numeric[:, attr_index] = np.log(
                    np.dot(bin_kernel, hist1) / np.dot(bin_kernel, hist0))
            else:
                attr_index = self.x_types_index[d]
                num_of_category = len(np.unique(self.x_tr[:, d]))
                # ヒストグラムの作成
                hist0 = np.bincount(self.x_tr[:, d][np.where(self.teach_labels == 0)].astype('int64'),
                                    minlength=num_of_category + 1)
                hist1 = np.bincount(self.x_tr[:, d][np.where(self.teach_labels == 1)].astype('int64'),
                                    minlength=num_of_category + 1)
                # 特徴量(phi)の計算
                self.feature_vectors_for_category[attr_index] = np.log((hist1 + 1) / (hist0 + 1))

    def get_bin_index_for_numerical(self, x_vectors_numerical):
        bin_index_for_numerical = np.ceil((x_vectors_numerical - self.min_value) / self.bin_length)
        # 学習データより大きいものは100にし、学習データより小さいものは1にする
        bin_index_for_numerical[np.where(bin_index_for_numerical > 100)] = 100
        bin_index_for_numerical[np.where(bin_index_for_numerical < 0)] = 0
        return bin_index_for_numerical

    def numeric_to_feature(self, x_vectors_numerical):
        bin_index_for_numerical = self.get_bin_index_for_numerical(x_vectors_numerical)
        num_samples, dim_features = x_vectors_numerical.shape
        feature_vectors_numerical = np.empty((num_samples, dim_features))
        for d in np.arange(dim_features):
            for i, index in enumerate(bin_index_for_numerical[:, d]):
                feature_vectors_numerical[i][d] = self.feature_vectors_for_numeric[index - 1][d]
        return feature_vectors_numerical

    def category_to_feature(self, x_vectors_categorical):
        num_samples, dim_features = x_vectors_categorical.shape
        feature_vectors_categorical = np.empty((num_samples, dim_features))
        for d in np.arange(dim_features):
            for i, index in enumerate(x_vectors_categorical[:, d]):
                feature_vectors_categorical[i][d] = self.feature_vectors_for_category[d][index]
        return feature_vectors_categorical

    def phi(self, x_vectors):
        x_vectors_numerical = x_vectors[:, self.numerical_index]
        x_vectors_categorical = x_vectors[:, self.categorical_index]
        feature_vectors_numerical = self.numeric_to_feature(x_vectors_numerical)
        feature_vectors_categorical = self.category_to_feature(x_vectors_categorical)
        feature_vectors = np.empty((self.num_samples, self.dim_features))
        feature_vectors[:, self.numerical_index] = feature_vectors_numerical
        feature_vectors[:, self.categorical_index] = feature_vectors_categorical
        feature_vectors = np.hstack((feature_vectors, np.ones(x_vectors.shape[0])[:, np.newaxis]))
        return feature_vectors

    def fit(self, x_vectors, teach_labels, x_types, method='newton'):
        self.x_tr = x_vectors
        self.num_samples, self.dim_features = self.x_tr.shape
        self.w_vector = np.random.rand(self.dim_features + 1)
        self.teach_labels = teach_labels
        self.x_types = x_types
        # 数値データ、カテゴリカルデータに応じたインデックスを作成
        self.set_x_types_index()
        self.method = method
        self.initialize_bandwidth_vector()
        self.initialize_num_of_bins()
        self.create_bin_model()
        self.estimate_weight()


if __name__ == '__main__':
    np.random.seed(0)
    X_num = np.random.randn(100, 3)
    X_cat = np.random.randint(1, 5, size=(100, 3))
    X = np.hstack((X_num, X_cat))


    def f(x):
        return 5 * x[0] - 4 * x[1] + 2 * x[2] - 2 * x[3] + 4 * x[4] - 3 * x[5]

    T = np.array([1 if f(x) > 0 else 0 for x in X])
    x_types = np.array([0, 0, 0, 1, 1, 1])

    model = DensityBasedLogisticRegressionWithBins()
    model.fit(X, T, x_types)
    print(model.w_vector)