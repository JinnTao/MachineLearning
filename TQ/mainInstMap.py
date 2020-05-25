# -*- coding:utf-8 -*-
# 主力合约映射
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqsdk
import time
api = tqsdk.TqApi()
inst = "CFFEX.IF"
mainInst = "KQ.m@" + inst
# 获取全部历史相关数据
totalInst = [x for x in api._data.quotes if inst == x[:len(inst)]]
# 获取主连行情
mainInstQuotes = api.get_kline_serial(mainInst,60*60*24,1000)
#得到主连成交量 dict
d = {mainInstQuotes.datetime.iloc[x] : mainInstQuotes.volume.iloc[x] for x in range(1000) if mainInstQuotes.datetime.iloc[x] > 0}
#获取全部历史相关的合约行情
totalInstQuotes = [api.get_kline_serial(x,24*60*60,1000) for x in totalInst]
#得到全部合约 {时间：(成交量：合约名字)}
l=[]
for x in totalInstQuotes:
    d1 = {}
    for y in range(1000):
        if x.datetime.iloc[y] > 0:
            d1[x.datetime.iloc[y]] = (x.volume.iloc[y], x.symbol.iloc[y])
    l.append(d1)

# 遍历对比相同时间，成交量相同，就把合约名字和时间形成字典
newd = {}
for x in d:
    for y in l:
        if x in y:
            if d[x] == y[x][0]:
                newd[x] = y[x][1]

#寻找相同值，但索引最小的
d = {}
for x in newd:
    if newd[x] not in d:
        d[newd[x]] = x
    else:
        if x < d[newd[x]]:
            d[newd[x]] = x

#将时间为键，合约名字为字典，写入记事本
fd = {}
for x in d:
    a = time.localtime(d[x]/1e9)
    b = time.strftime("%Y-%m-%d",a)
    fd[b] = x

f = open("InstMap/"+inst+"_switch.txt",'w')
f.write(str(fd))
f.close()

#每日是那个品种写入记事本
fd = {}
for x in newd:
    a = time.localtime(x/1e9)
    b = time.strftime("%Y-%m-%d",a)
    fd[b] = newd[x]


f = open("InstMap/"+inst+"_daily.txt",'w')
f.write(str(fd))
f.close()
api.close()

