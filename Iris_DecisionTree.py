# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pydotplus

iris_features_E = 'sepal_length', 'sepal width', 'pedal length', 'pedal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

if __name__ == '__main__':
    # 让 matplot 图片支持中文
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    data = pd.read_csv('Data/iris.data', header=None)
    x = data[range(4)]
    y = pd.Categorical(data[4]).codes
    # 为了可视化使用两列特征值
    x = x.iloc[:, :2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
    # 决策树参数设计
    model = DecisionTreeClassifier(criterion='entropy',max_depth=10)
    model.fit(x_train, y_train)
    y_test_hat = model.predict(x_test)

    # 保存
    # 1. 输出
    with open('Data/iris.dot', 'w') as f:
        tree.export_graphviz(model, out_file=f)
    f.close()

    # 2.给定文件名
    # tree.export_graphviz(model,out_file='iris1.dot')
    # 3.输出为pdf
    dot_data = tree.export_graphviz(model, out_file=None, feature_names=iris_features_E[:2], class_names=iris_class,
                                    filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('Data/iris.pdf')
    f = open('Data/iris.png', 'wb')
    f.write(graph.create_png())
    f.close()

    # 画图
    N, M = 50, 50
    x1_min, x2_min= x.min()
    x1_max, x2_max = x.max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)
    x_show = np.stack((x1.flat, x2.flat), axis=1)
    print(x_show.shape)

    # # 无意义 只是为了凑另外两个维度
    # 打开注释前，确保注释掉 x = x[:, :2]
    # x3 = np.ones(x1.size) * np.average(x.iloc[:, 2])
    # x4 = np.ones(x1.size) * np.average(x.iloc[:, 3])
    # x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1)

    cm_light = mpl.colors.ListedColormap(['#A0FFA0','#FFA0A0','#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g','r','b'])
    y_show_hat = model.predict(x_show)
    print (y_show_hat.shape)
    print (y_show_hat)
    y_show_hat = y_show_hat.reshape(x1.shape) # 让与输入形状相同
    print (y_show_hat)
    plt.figure(facecolor='w')
    plt.pcolormesh(x1,x2,y_show_hat,cmap=cm_light)
    plt.scatter(x_test[0], x_test[1], c=y_test.ravel(),edgecolors='k',s=150,zorder= 10,cmap=cm_dark,marker='*')
    plt.scatter(x_train[0],x_train[1],c=y_train.ravel(),edgecolors='k',s=40,cmap=cm_dark)
    plt.xlabel(iris_feature[0],fontsize=15)
    plt.ylabel(iris_feature[1], fontsize=15)
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(True)
    plt.title(u'莺尾花数据决策树分类',fontsize=17)
    plt.show()

    #训练集上的预测结果
    y_test = y_test.reshape(-1)
    print (y_test_hat.shape)
    print(y_test.shape)
    result = (y_test_hat == y_test)
    acc = np.mean(result)
    print('准确度: %.2f%%' % (100*acc))
    #过拟合  错误率
    depth = np.arange(1,15)
    err_list=[]
    for d in depth:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        clf.fit(x_train,y_train)
        y_test_hat = clf.predict(x_test)
        result = (y_test_hat == y_test)
        err = 1 - np.mean(result)
        err_list.append(err)
        print (d , '错误率: %.2f%%' % (100*err))
    plt.figure(facecolor='w')
    plt.plot(depth,err_list,'ro-',lw=2)
    plt.xlabel(u'决策树深度',fontsize=15)
    plt.xlabel(u'错误率',fontsize=15)
    plt.title(u'决策树深度与过拟合',fontsize=17)
    plt.grid(True)
    plt.show()


