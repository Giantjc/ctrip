# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 00:04:14 2017
计算商品之间的两两相似度
CalculatSim()函数计算得到datasim矩阵
@author: giantjc
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans 
from scipy import sparse
from itertools import *  
  
#计算两个商品的相似度
def similarity(item1,item2):
    catergory = ['district_id1','district_id2','district_id3']
    numerical = ['district_id4',#'lat','lon','railway','airport','citycenter','railway2','airport2','citycenter2',
                 'eval','eval2','eval3','eval4','voters','maxstock']
    n = 0; count = 0; sum1 = 0
    for cols in catergory:
        if item1[cols] == -1 or item2[cols]== -1:
            continue;
        count = count + 1
        if item1[cols] == item2[cols]:           
            n = n+1
    sim1 = float(1-n/3.0)     
    for cols in numerical:
        if item1[cols] == -1 or item2[cols]==-1:
            continue;
        count = count+1
        sum1 += np.square(item1[cols]-item2[cols])
    sim2 = np.sqrt(sum1)
    return (sim1+sim2)/count

#对商品属性进行K均值聚类
def k_means_cluster(data,K):  
    x_p = data.iloc[:, 0:2] # 取前2列
    clf = KMeans(n_clusters=K,  n_init=1, verbose=1)  
    clf.fit(x_p)  
    labels = clf.labels_#样本点被分配到的簇的索引
    label = pd.DataFrame(list(clf.labels_),columns=['LABELS'])
    frame = [x_p,label]
    x_new = pd.concat(frame,axis=1)  
    return x_new    

#线性扫描，暴力求两两的相似度
def CalculateSim(product_info):
    list1 = []
    product_info_union = product_info
    combination = combinations(product_info_union.index,2)   
    for index_a,index_b in combination:
        sim = similarity(product_info_union.loc[index_a],product_info_union.loc[index_b])
        list1.append([index_a,index_b,sim])        
    datasim = pd.DataFrame(list1,columns=['pro1','pro2','sim'])
    return datasim



      