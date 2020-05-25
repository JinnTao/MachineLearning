# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt
import os
from PIL import Image


def update(f):
    global loc
    if f == 0:
        loc = loc_prime
    next_loc = np.zeros((m,n),dtype=np.float)
    for i in np.arange(m):
        for j in np.arange(n):
            next_loc[i,j] = calc_next_loc(np.array([i,j]),loc,directions)
    loc = next_loc / np.max(next_loc)
    im.set_array(loc)

    if save_image:
        if f%3 == 0:
            image_data = plt.cm.coolwarm(loc) * 255
            image_data, _ =np.split(image_data, (-1,),axis=2)
            image_data = image_data.astype(np.uint8).clip(0,255)
            output = ".\\Pic2\\"
            if not os.path.exists(output):
                os.mkdir(output)
            a = Image.fromarray(image_data,mode='RGB')
            a.save('%s%d.png'%(output,f))
    return [im]
def calc_next_loc(now,loc,directions):
    near_index = np.array([(-1,-1),(-1,0),(-1,1),(0,-1),
                           (0,1),(1,-1),(1,0),(1,1)])
    directions_index = np.array([7,6,5,0,4,1,2,3]) # 这个排列顺序为什么是这样子的？
    nn = now + near_index
    ii,jj = nn[:,0] , nn[:,1]
    ii[ii>=m] = 0
    jj[jj >= n] = 0
    return np.dot(loc[ii,jj],directions[ii,jj,directions_index])


if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=300,edgeitems=8)
    np.random.seed(0)

    save_image = False
    style = 'Sin'
    m,n= 50,100
    directions = np.random.rand(m,n,8)

    if style == "Direct":
        directions[:,:,1] = 10
    elif style == 'Sin':
        x = np.arange(n)
        y_d = np.cos(6 * np.pi * x /n) # 3 个余弦震荡
        theta = np.empty_like(x,dtype=np.int)
        theta[y_d > 0.5] = 1
        theta[~(y_d > 0.5) & (y_d > -0.5)] = 0
        theta[~(y_d > -0.5)] = 7
        print (theta)
        # 对应位置坐标设置为对应值10，如下 d[:,[a1,a2],[b1,b2]],每个块的[a1,b1] [a2,b2] 设置为10
        directions[:,x.astype(np.int),theta] = 10
    directions[:,:] /= np.sum(directions[:,:])

    loc = np.zeros((m,n),dtype=np.float)
    loc[int(m/2),int(n/2)]=1
    loc_prime = np.empty_like(loc,dtype=np.float)
    loc_prime = loc
    fig = plt.figure(figsize=(8,6),facecolor='w')
    im = plt.imshow(loc/np.max(loc),cmap='coolwarm')
    anim = animation.FuncAnimation(fig,update,frames=300,interval=50,blit=True)
    plt.tight_layout(1.5)
    plt.show()

    # print (directions)


