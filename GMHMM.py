# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from hmmlearn import hmm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import accuracy_score
import warnings


def expand(a, b):
    return 1.05 * a - 0.05 * b, 1.05 * b - 0.05 * a


if __name__ == "__main__":
    warnings.filterwarnings('ignore')  # hmm(0.2)
    np.random.seed(0)

    n = 5
    n_samples = 500
    pi = np.random.rand(n)
    pi /= pi.sum()
    print('初始概率:', pi)

    A = np.random.rand(n, n)
    mask = np.zeros((n, n), dtype=np.bool)
    mask[0, 1] = mask[0, 4] = True
    mask[1, 0] = mask[1, 2] = True
    mask[2, 1] = mask[2, 3] = True
    mask[3, 2] = mask[3, 4] = True
    mask[4, 0] = mask[4, 3] = True
    A[mask] = 0
    A = A / A.sum(axis=1)[:, np.newaxis]  # 广播 归一化

    means = np.array(((30, 30, 30), (0, 50, 20), (-25, 30, 10), (-15, 0, 25), (15, 0, 40)), dtype=np.float)
    # 归一化 欧式距离
    means = means / np.sqrt(np.sum(means ** 2, axis=1))[:, np.newaxis]
    # way 1
    covars = np.array([np.diag(np.random.rand(3) * 0.03 + 0.001) for _ in np.arange(n)])
    # way 2
    # covars = np.empty((n,3,3))
    # for i in range(n):
    #     #covars[i] = np.diag(np.random.randint(1,5,size=3))
    #     covars[i] = np.diag(np.random.rand(3)*0.03 + 0.001)
    print(covars)

    model = hmm.GaussianHMM(n_components=n, covariance_type='full')
    model.startprob_ = pi
    model.transmat_ = A
    model.means_ = means
    model.covars_ = covars
    # 采样，这里的Sample等同于X,labels等同于Y
    sample, labels = model.sample(n_samples=n_samples, random_state=0)

    # 估计参数 无监督学习 不使用Labels
    model = hmm.GaussianHMM(n_components=n, covariance_type='full', n_iter=10)
    model.fit(sample)
    y = model.predict(sample)
    np.set_printoptions(suppress=True)
    # 下列输出没有方向调整的
    print("##估计初始概率:", model.startprob_)
    print("##估计转移概率:", model.transmat_)
    print("##估计均值:", model.means_)
    print("##估计方差:", model.covars_)

    # 类别 方向调整
    order = pairwise_distances_argmin(means, model.means_, metric="euclidean")
    print(order)
    pi_hat = model.startprob_[order]
    A_hat = model.transmat_[order]  # 行反转
    A_hat = A_hat[:, order]  # 列翻转
    means_hat = model.means_[order]
    covars_hat = model.covars_[order]
    change = np.empty((n, n_samples), dtype=np.bool)
    # 原始顺序 0 1 2 3 4
    # 新的顺序 order 由于没有方向调整，所以按order重新排列，给上标记值
    for i in range(n):
        change[i] = (y == order[i])
    for i in range(n):
        y[change[i]] = i
    print("估计初始转移概率:", pi_hat)
    print("估计转移概率：", A_hat)
    print("估计均值：", means_hat)
    print(labels)
    print("准确率：%.2f%%" % accuracy_score(labels, y))

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(8, 8), facecolor='w')
    ax = fig.add_subplot(111, projection='3d')
    #colors = plt.cm.Spectral(np.linspace(0,1,n))
    ax.scatter(sample[:,0],sample[:,1],sample[:,2],s=5,c=labels,cmap=plt.cm.Spectral)
    plt.plot(sample[:, 0], sample[:, 1], sample[:, 2], lw=0.1, color="#A07070")
    colors = plt.cm.Spectral(np.linspace(1, 0, n))
    ax.scatter(means[:,0],means[:,1],means[:,2],s=300,c=colors,edgecolor='r',linewidth=0.1)

    x_min, y_min, z_min = sample.min(axis=0)
    x_max, y_max, z_max = sample.max(axis=0)
    x_min, x_max = expand(x_min, x_max)
    y_min, y_max = expand(y_min, y_max)
    z_min, z_max = expand(z_min, z_max)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    ax.set_zlim((z_min, z_max))
    # plt.legend(loc='upper left')
    plt.grid(True)
    # plt.tight_layout(1)
    plt.title(u'GMHMM 参数估计和类别判定', fontsize=18)
    plt.show()
