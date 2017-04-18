# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 18:34:15 2017
进行数据预处理工作的函数
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

#%%
#对product_info的类别字段进行处理
def trans2label(cols,col):
    le = preprocessing.LabelEncoder()
    le.fit(cols) 
    cols_transf = list(le.transform(cols))
    da = pd.DataFrame(data = cols_transf,index = cols.index,columns = {col})
    return da

#处理时间类型，时间字符串转换为间隔
def trans2time(time1):
    if time1 == '1753-01-01' or time1 == '-1':
        return -1
    basedate = datetime(1753,1,1)
    tdate = datetime.strptime(time1, "%Y-%m-%d")  
    deltaday = (tdate - basedate).days   
    return deltaday

#处理train_month的缺失值，前提是有datasim数据
def fillnan(train_month,cnt,topK,fillnum):
    null_index = nullcount(train_month,cnt)#缺失值大于cnt个月的
    for index in null_index:
        single_sim = datasim[(datasim.pro1==index)|(datasim.pro2==index)].sort_values(by=['sim'])
        sim_index=(set(single_sim.head(topK).pro1)|set(single_sim.head(topK).pro2))-{index}            
        for month in train_month.columns:
            if np.isnan(train_month.loc[index,month]):
                train_month.loc[index,month] = train_month.loc[sim_index,month].mean(axis=0)
    train_month.fillna(fillnum,inplace=True) 
    return train_month 

#对product_info 进行数据预处理, 数据预处理 + 标准化，统一不同属性分量的大小不一致问题  
def preprocess(product_info):   
    for col in ['startdate','upgradedate','cooperatedate']:
        product_info[col] = product_info[col].apply(lambda x: trans2time(x))
    '''
    scaler = preprocessing.MinMaxScaler()
    x_train = scaler.fit_transform(product_info.values)
    product_info_union = pd.DataFrame(x_train,index=product_info.index,columns=product_info.columns)
    '''
    #hahaha = pd.concat(map(lambda x: trans2label(product_info.loc[:,x],x),product_info.columns[:3]),axis=1) 
    return product_info
#%%
#统计每个商品为缺失值的月份
def nullcount(train_month,count):
    null = train_month.apply(lambda x:sum(x.isnull()),axis=1)
    null_index = null[null >= count].index
    return null_index

#基于相似填充nan值
def myfillna(train_month,nullindex):
    for index in nullindex:
        mid = train_month.loc[index].dropna().median()
        avg = train_month.loc[index].dropna().mean()
        fill = (mid + avg)/2
        train_month.loc[index].fillna(fill,inplace = True)
    else:
        train_month.loc[index].fillna(140,inplace = True)
    return train_month

#把数据变为正数
def to_positive(result):
    #把预测的负数改变为 0
    for index,row in result.iterrows():
        for col_name in result.columns:
            if row[col_name] < 0:
                row[col_name] = 0
    return result
#%%