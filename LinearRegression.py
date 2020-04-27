# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    data = pd.read_csv('data/Advertising.data')
    x = data[['TV','radio','newspaper']]
    y = data['sales']

    x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=1)
    model = Lasso()
    model = Ridge()

    alpha_can = np.logspace(-3,2,base=10)
    np.set_printoptions(suppress = True)
    print('alpha_can: {}'.format(alpha_can))
    # find the best alpha of L1 norm
    lasso_model = GridSearchCV(model,{'alpha':alpha_can},cv = 5)
    lasso_model.fit(x_train,y_train)
    print('best alpha : {}'.format(lasso_model.best_params_))

    order = y_test.argsort(axis = 0)
    y_test = y_test.values[order]
    x_test = x_test.values[order]
    y_hat = lasso_model.predict(x_test)
    print (lasso_model.score(x_test,y_test))

    mse = np.average((y_hat - np.array(y_test)) ** 2)
    rmse = np.sqrt(mse)
    print ('mse : {}, rmse: {}'.format(mse, rmse))

    t = np.arange(len(x_test))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(facecolor='w')
    plt.plot(t,y_test,'r-',linewidth=2,label = 'True Data')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict Data')
    plt.title(u'线性回归预测销售数据', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

    print('best')





