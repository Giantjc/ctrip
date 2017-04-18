# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 00:04:14 2017
对product_info进行聚类，填充nan值
@author: giantjc
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
import matplotlib.pyplot as plt
import csv
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
    #cents = clf.cluster_centers_#质心
    labels = clf.labels_#样本点被分配到的簇的索引
    #print(clf.cluster_centers_) 
    # print(list(clf.labels_))
    label = pd.DataFrame(list(clf.labels_),columns=['LABELS'])
    frame = [x_p,label]
    x_new = pd.concat(frame,axis=1)  
    return x_new    
#%%
#线性扫描，暴力求两两的相似度
list1 = []
combination = combinations(product_info.index,2)
for index_a,index_b in combination:
    sim = similarity(product_info_union.loc[index_a],product_info_union.loc[index_b])
    list1.append([index_a,index_b,sim])
    
datasim = pd.DataFrame(list1,columns=['pro1','pro2','sim'])
#%%
#构造KD树，寻找每个样本的K个近邻
      
#%%
#对于完全没有数据的500多个样本的缺失值进行处理
totol_null_index = set(product_info.index)-set(train_month.index)#完全缺失值
for index in totol_null_index:
    single_sim = datasim[(datasim.pro1==index)|(datasim.pro2==index)].sort_values(by=['sim'])
    sim_index=(set(single_sim.head(10).pro1)|set(single_sim.head(10).pro2))-{index} 
    count = 0; sum1=list(np.zeros(14))
    #sum1 = list(predict_month.loc[sim_index].mean(axis=0))    
    for i in sim_index:
        if i not in totol_null_index:
            count += 1
            sum_tmp = list(out.loc[out.product_id == i,out.columns[2]])
            sum1 = list(map(lambda x: (x[0] + x[1]), zip(sum_tmp,sum1)))
    sum1 = [ii/count for ii in sum1]
    
    out.loc[out.product_id == index,out.columns[2]] = sum1 #根据相似的商品预测值填补缺失值      
    '''
    for index in sim_index:
        train_month[train_month.index == index].sum().plot(figsize=(12,6))
    '''