# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 21:08:10 2017

@author: giantjc
"""
import sys
import os
pre_path = os.path.abspath('../')
sys.path.append(pre_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cluster as clt
import data_preprocessing as prep
import train_function as trainf

#读取数据，简单预处理
product_info = pd.read_csv("product_info.txt",index_col='product_id')
product_quantity = pd.read_csv("product_quantity.txt",)
product_quantity.sort_values(['product_id','product_date'],inplace=True)
train_day=product_quantity.groupby(['product_id','product_date']).sum()['ciiquantity'].unstack()

train_day.fillna(method='backfill',axis=1) 
train_day.fillna(method='ffill',axis=1) 

product_quantity['product_month']=product_quantity['product_date'].apply(lambda x: x[:7])
train_month = product_quantity.groupby(['product_id','product_month']).sum()['ciiquantity'].unstack()
total_null_index = set(product_info.index)-set(train_month.index)#完全缺失值的index

#计算相似性矩阵
#data_sim = clt.CalculateSim(product_info)  #计算时间较长，可直接将datasim导入Variable explore
datasim = pd.read_csv("datasim.csv",index_col='Index') #直接读入，用于填充缺失值
#缺失值填补（均值填补）
train_month = prep.fillnan(train_month,1,30,140,datasim)#对大于12个月有空缺值的基于相似度进行填补，其他的填补为140，取topK=30进行填补
#对product_info进行数据预处理
product_info = prep.preprocess(product_info) 
#计算出每个产品的平均价格
product_quantity['price'].replace(to_replace=-1,method='backfill',inplace=True)
product_quantity['price'].replace(to_replace=-1,method='ffill',inplace=True) 
product_price = product_quantity.groupby(['product_id']).mean()['price']

#对多个ID同时进行训练,采用xgboost
result_for_some = pd.DataFrame()
result_for_some,feature_importance = trainf.train_for_some(train_month,product_info,train_month.index,product_price,'xgboost')
result_for_some = prep.to_positive(result_for_some.iloc[:,23:])

#采用均值的方法
average = pd.DataFrame(train_month.iloc[:,14:23].mean(axis=1),columns=['average_all']).reset_index()
submission=pd.read_csv('sample.txt')
col=['product_id','product_month','ciiquantity_month']
submission.columns=col
out=pd.merge(submission,average,on='product_id',how='left')
out.ciiquantity_month=out.average_all
out.drop(['average_all'],axis=1,inplace=True)

#在均值的基础上融合XGBOOST模型预测（横向）
for index in result_for_some.index:
    v2 = list(result_for_some.loc[index,:])
    v1 = list(out.loc[out.product_id==index,out.columns[2]])
    v = list(map(lambda x: (x[0]+x[1])/2, zip(v2, v1)))
    out.loc[out.product_id==index,out.columns[2]] = v
out = prep.fillTotalNan(out,28,total_null_index,datasim)
out2 = pd.read_csv('regulation.txt') #规则得到的模型
out['ciiquantity_month'] = (out2['ciiquantity_month']+out['ciiquantity_month'])/2

#采用随机森林算法并进行融合
result_for_some = pd.DataFrame()
result_for_some,feature_importance = trainf.train_for_some(train_month,product_info,train_month.index,product_price,'rf')
result_for_some = prep.to_positive(result_for_some.iloc[:,23:])
for index in result_for_some.index:
    v2 = list(result_for_some.loc[index,:])
    v1 = list(out.loc[out.product_id==index,out.columns[2]])
    v = list(map(lambda x: (x[0]+x[1]*3)/4, zip(v2, v1)))
    out.loc[out.product_id==index,out.columns[2]] = v
           
#采用ExtraTrees算法并进行融合
result_for_some = pd.DataFrame()
result_for_some,feature_importance = trainf.train_for_some(train_month,product_info,train_month.index,product_price,'et')
result_for_some = prep.to_positive(result_for_some.iloc[:,23:])
for index in result_for_some.index:
    v2 = list(result_for_some.loc[index,:])
    v1 = list(out.loc[out.product_id==index,out.columns[2]])
    v = list(map(lambda x: (x[0]+x[1]*4)/5, zip(v2, v1)))
    out.loc[out.product_id==index,out.columns[2]] = v
           
#采用gbdt算法并进行融合
result_for_some = pd.DataFrame()
result_for_some,feature_importance = trainf.train_for_some(train_month,product_info,train_month.index,product_price,'gbdt')
result_for_some = prep.to_positive(result_for_some.iloc[:,23:])
for index in result_for_some.index:
    v2 = list(result_for_some.loc[index,:])
    v1 = list(out.loc[out.product_id==index,out.columns[2]])
    v = list(map(lambda x: (x[0]+x[1]*5)/6, zip(v2, v1)))
    out.loc[out.product_id==index,out.columns[2]] = v
           
out.to_csv('final_result.txt',index=False)


