# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats

def poly_regression():
    N = 50
    x = np.linspace(0,100,50)
    y = x * 3 + 5*np.random.randn(N)
    model = np.polyfit(x,y,deg=5)
    p1 = np.poly1d(model)
    plt.plot(x,y)
    plt.show()
    print (p1)
def test():
    res = []
    w_list = []
    for _ in range(0,10000):
        nor = pd.Series(np.random.randn(1000))
        w,p = stats.shapiro(nor)
        print('w:',w, 'p:',p)
        w_list.append(w_list)
        res.append(p)
    plt.figure()
    plt.hist(res,bins=1000)
    plt.show()
    plt.figure()
    plt.hist(w_list)
    plt.show()

def cos():
    x = np.arange(600)
    y =np.cos( 6*x * np.pi / 600.)
    plt.figure(facecolor='w')
    plt.plot(x,y)
    plt.show()
if __name__ == '__main__':
    cos()