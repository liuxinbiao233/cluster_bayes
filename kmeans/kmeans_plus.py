import time
import pandas as pd
from sklearn.cluster import KMeans

from sklearn import datasets

import numpy as np

from utils.misc_utils import distance, sortLabel
from kmeans.kmeans_base import KMeansBase
import matplotlib.pyplot as plt
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs


class KMeansPlusPlus(KMeansBase):

    def __init__(self, n_clusters = 8, init="random", max_iter = 300, random_state = None, n_init = 10, tol = 1e-4):
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





    df1 = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\cluster_data.csv', usecols=[1,2,3])
    data = df1.values
    iris = datasets.load_iris()
    startTime = time.time()
    # km1 = KMeansBase(3)

    # labels = km1.fit_predict(iris.data)
    # print("km1 time",time.time() - startTime)
    # print(np.array(sortLabel(labels)))
    #
    # km2 = KMeansPlusPlus(3, init="k-means++")
    # startTime = time.time()
    # labels = km2.fit_predict(data)
    # print("km2 time", time.time() - startTime)
    # print(np.array(sortLabel(labels)))

    kmeans= KMeans(init='k-means++', n_clusters= 3,)
    startTime = time.time()
    label = kmeans.fit(data)
    print(kmeans.cluster_centers_)

    # times = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\0_times.csv', usecols=[1])
    # a = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\acceleration_mean.csv', usecols=[1])
    # v = pd.read_csv(r'C:\Users\DaBiao\Downloads\PyCharmProject\data_test\v_mena.csv', usecols=[1])
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # ax = plt.subplot(projection='3d')
    # x = v.values[:, 0],
    # y = a.values[:, 0],
    # z = times.values[:, 0]

    # colors = ['#4EACC5', '#4E9A06', 'm', '#FF9C34']
    #
    # ax.scatter(v.values[label == 0][:, 0], a.values[label == 0][:, 0], times.values[label == 0][:, 0], c='#4EACC5',
    #            marker='.', alpha=0.7, s=10)
    # ax.scatter(v.values[label == 1][:, 0], a.values[label == 1][:, 0], times.values[label == 1][:, 0], c='#4E9A06',
    #            marker='.', alpha=0.7, s=10)
    # ax.scatter(v.values[label == 2][:, 0], a.values[label == 2][:, 0], times.values[label == 2][:, 0], c='#FF9C34',
    #            marker='.', alpha=0.7, s=10)
    # ax.scatter(v.values[label == 3][:, 0], a.values[label == 3][:, 0], times.values[label == 3][:, 0], c='m',
    #            marker='.', alpha=0.7, s=10)
    #
    # plt.title("组合工况片段聚类三维散点图")
    # ax.set_xlabel('平均速度vi_mean/(km/h)')
    # ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
    # ax.set_ylabel('平均绝对加速度(m/s^2)')
    # ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    # ax.set_zlabel('怠速时间比')
    # ax.set_zticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # plt.figure(dpi=1000, figsize=(65, 65))
    #
    # ax.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],c='r',s=35,marker='*')
    # plt.show()


    # # ax.scatter(kmeans.cluster_centers_[], c='r', s=25, marker='*')
    # # ax.scatter(kmeans.cluster_centers_[:, 2], c='r', s=25, marker='*')
    # # ax.scatter(kmeans.cluster_centers_[:, 3], c='r', s=25, marker='*')
    # print(kmeans.cluster_centers_[:,0])
    # print(kmeans.cluster_centers_[:,1])
    # print(kmeans.cluster_centers_[:,2])






    # print("sklearn time",time.time() - startTime)

    print("出现0的次数82次，概率为0.3166，出现1的次数63次，概率为0.2432，出现2的次数114次概率为0.4401")

    # print(kmeans.inertia_)

    #肘部法
    # sse = []
    # scope = range(1, 10)
    # for k in scope:
    #     kmeans = KMeans(n_clusters=k)
    #     kmeans.fit(data)
    #     # SSE inertia_属性
    #     sse.append(kmeans.inertia_)
    # plt.xticks(scope)
    # plt.plot(scope, sse, marker="o")


