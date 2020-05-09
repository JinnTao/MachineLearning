# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.tree import

iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == '__main__':
    # 让 matplot 图片支持中文
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    # load Data
    data = pd.read_csv('Data/iris.data', header=None)
    x, y = data[range(4)],data[4]
    y = pd.Categorical(y).codes
    x = x[[0,1]]

    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=1,train_size=0.6)

    #分类器
    #clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf = svm.SVC(C=0.8, kernel='rbf', decision_function_shape='ovr',gamma='scale')
    clf.fit(x_train, y_train.ravel())

    # accurate rate
    print (clf.score(x_train,y_train))
    print (clf.score(x_test,y_test))
    print (accuracy_score(y_train,clf.predict(x_train)))
    print (accuracy_score(y_test,clf.predict(x_test)))

    # draw pic
    x1_min, x2_min = x.min()
    x1_max, x2_max = x.max()
    x1,x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    grid_test = np.stack((x1.flat,x2.flat),axis=1)
    grid_hat = clf.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)

    cm_light = mpl.colors.ListedColormap(['#A0FFA0','#FFA0A0','#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g','r','b'])
    plt.figure(facecolor='w')
    plt.pcolormesh(x1,x2,grid_hat,cmap=cm_light)
    #plt.show()
    plt.scatter(x[0], x[1], c=y,edgecolors='k',s=50,cmap=cm_dark,marker='*')
    plt.scatter(x_test[0], x_test[1],c=clf.predict(x_test),edgecolors='k',s=50,cmap=cm_dark)
    plt.xlabel(iris_feature[0],fontsize=13)
    plt.ylabel(iris_feature[1], fontsize=13)
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'SVM classification', fontsize=16)
    plt.grid(b=True,ls = ':')
    plt.tight_layout(pad=1.5)
    plt.show()


