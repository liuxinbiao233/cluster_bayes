import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import pandas as pd
from sklearn.cluster import kmeans_plusplus
from sklearn.naive_bayes import GaussianNB

import time
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import datasets


import numpy as np

from utils.misc_utils import distance, sortLabel
from kmeans.kmeans_base import KMeansBase
import matplotlib.pyplot as plt
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
import pandas as pd


class GaussianNaiveBayes(object):
    def __init__(self):
        self.model = None

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        return norm.pdf(x, loc=mean, scale=stdev)

    # 处理X_train
    def summarize(self, train_data):
        summaries = [(np.mean(X), np.std(X)) for X in zip(*train_data)]
        return summaries

    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        labels = list(set(y))
        data = {label:[] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {label: self.summarize(value) for label, value in data.items()}
        return self

    # 计算概率
    def calculate_probabilities(self, input_data):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
        return probabilities

    # 类别
    def predict(self, X_test):
        label = sorted(self.calculate_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1

        return right / float(len(X_test))

class KMeansPlusPlus(KMeansBase):

    def __init__(self, n_clusters=8, init="random", max_iter=300, random_state=None, n_init=10, tol=1e-4):
        super(KMeansPlusPlus, self).__init__(
            n_clusters=n_clusters, init=init, max_iter=max_iter,
            random_state=random_state, tol=tol, n_init=n_init)

    def _init_centroids(self, dataset):
        n_samples = dataset.shape[0]
        centers = []
        if self.init == "random":
            seeds = self.random_state.permutation(n_samples)[:self.k]
            centers = dataset[seeds]
        elif self.init == "k-means++":
            centers = self._k_means_plus_plus(dataset)
        return np.array(centers)

    # kmeans++的初始化方式，加速聚类速度
    def _k_means_plus_plus(self, dataset):
        n_samples, n_features = dataset.shape
        centers = np.empty((self.k, n_features))
        # n_local_trials是每次选择候选点个数
        n_local_trials = None
        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(self.k))

        # 第一个随机点
        # center_id = self.random_state.randint(n_samples)
        center_id = self.random_state.randint(n_samples)
        centers[0] = dataset[center_id]

        # closest_dist_sq是每个样本，到所有中心点最近距离
        # 假设现在有3个中心点，closest_dist_sq = [min(样本1到3个中心距离),min(样本2到3个中心距离),...min(样本n到3个中心距离)]
        closest_dist_sq = distance(centers[0, np.newaxis], dataset)

        # current_pot所有最短距离的和
        current_pot = closest_dist_sq.sum()

        for c in range(1, self.k):
            # 选出n_local_trials随机址，并映射到current_pot的长度
            rand_vals = self.random_state.random_sample(n_local_trials) * current_pot
            # np.cumsum([1,2,3,4]) = [1, 3, 6, 10]，就是累加当前索引前面的值
            # np.searchsorted搜索随机出的rand_vals落在np.cumsum(closest_dist_sq)中的位置。
            # candidate_ids候选节点的索引
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)

            # best_candidate最好的候选节点
            # best_pot最好的候选节点计算出的距离和
            # best_dist_sq最好的候选节点计算出的距离列表
            best_candidate = None
            best_pot = None
            best_dist_sq = None
            for trial in range(n_local_trials):
                # 计算每个样本到候选节点的欧式距离
                distance_to_candidate = distance(dataset[candidate_ids[trial], np.newaxis], dataset)

                # 计算每个候选节点的距离序列new_dist_sq， 距离总和new_pot
                new_dist_sq = np.minimum(closest_dist_sq, distance_to_candidate)
                new_pot = new_dist_sq.sum()

                # 选择最小的new_pot
                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            centers[c] = dataset[best_candidate]
            current_pot = best_pot
            closest_dist_sq = best_dist_sq

        return centers




if __name__ == "__main__":
    df1 = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\cluster_data.csv', usecols=[1, 2, 3])
    data = df1.values


    km2 = KMeansPlusPlus(3, init="k-means++", n_init=20)
    startTime = time.time()
    labels = km2.fit_predict(data)
    X,y = data, labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(len(X_train))
    print(len(X_test))
    model = GaussianNB()

    # c1 = model.fit(X_train, y_train).predict(X_train)
    # print((y_train != c1).sum())
    # print(model.score(X_train, y_train))


    model.fit(X_train, y_train)
    y_pred=model.fit(X_train,y_train).predict(X_test)
    print(y_test!=y_pred)
    print("总共的的%d点,误判的点有%d个" % (X_test.shape[0], (y_test != y_pred).sum()))
    print(model.score(X_test, y_test))
    # print(y_pred[:])
    # print(y_test)
    pot={'y_test':y_test,'y_pred':y_pred[:]}
    df=pd.DataFrame(pot)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(figsize=(23, 13))
    axes.plot(df.values[:,0], '-', drawstyle='steps-post', label='真正路况',color='red')
    axes.plot(df.values[:, 1], '-.', drawstyle='steps-post', label='识别路况',color='green')
    xticks = []
    plt.yticks([0, 1, 2])
    for i in range(len(X_test)):
        xticks.append(i)
    plt.xticks(xticks)
    axes.legend(loc='upper right')







