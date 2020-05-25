# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,MinMaxScaler,PolynomialFeatures
# GaussianNB 连续性高斯朴素贝叶斯
# MultinomialNB 离散型朴素贝叶斯
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


# 配色搭配： bg:Pastel2 fore:Dark2
if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target
    data = pd.DataFrame(data=X,columns=iris.feature_names)
    features = [0,2]
    features_names = np.array(iris.feature_names)[features]
    X = X[:,features]
    x, x_test, y,y_test = train_test_split(X,Y,train_size=0.7,random_state=0)

    priors = np.array((1,2,4),dtype=float)
    priors = priors / priors.sum()
    gnb = Pipeline([
        ('sc',StandardScaler()),
        ('poly',PolynomialFeatures(degree=1)),
        ('clf',GaussianNB(priors=priors))
    ])
    #gnb = KNeighborsClassifier(n_neighbors=3).fit(x,y.ravel())
    gnb.fit(x,y.ravel())
    y_hat = gnb.predict(x)
    print ('训练集精度 %.2f%%' % (100*accuracy_score(y,y_hat)))
    y_test_hat = gnb.predict(x_test)
    print('训练集精度 %.2f%%' % (100 * accuracy_score(y_test, y_test_hat)))

    N,M = 500,500
    x1_min,x2_min = X.min(axis=0)
    x1_max,x2_max = X.max(axis=0)
    t1 = np.linspace(x1_min,x1_max,N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1,x2 = np.meshgrid(t1,t2)
    x_grid = np.stack((x1.flat,x2.flat),axis=1)
    y_grid_hat = gnb.predict(x_grid)

    y_grid_hat = y_grid_hat.reshape(x1.shape)


    plt.figure(facecolor='w')
    plt.pcolormesh(x1,x2,y_grid_hat,cmap=plt.cm.Pastel2)
    plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Dark2)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test,marker='^',s=120,edgecolors='k', cmap=plt.cm.Dark2)

    plt.xlabel(features_names[0],fontsize=13)
    plt.ylabel(features_names[1],fontsize=13)
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)
    plt.title(u'Gaussian NB分类',fontsize = 13)
    plt.grid(True)
    plt.show()



