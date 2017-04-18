# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 21:08:10 2017

@author: giantjc
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
#import train_function
#%%
#人工经验统计（节假日天数，周末天数(除节假日)，是否寒暑假）
knowledge = {
"2014-01":[2,7,1,0],"2014-02":[6,5,1,0],"2014-03":[0,10,0,0],"2014-04":[3,6,0,0],"2014-05":[4,6,0,0],"2014-06":[2,8,0,0],
"2014-07":[0,8,0,1],"2014-08":[1,10,0,1],"2014-09":[3,5,0,0],"2014-10":[7,5,0,0],"2014-11":[0,10,0,0],"2014-12":[0,8,0,0],
"2015-01":[3,7,1,0],"2015-02":[7,4,1,0],"2015-03":[0,9,0,0],"2015-04":[3,6,0,0],"2015-05":[3,8,0,0],"2015-06":[3,6,0,0],
"2015-07":[0,8,0,1],"2015-08":[1,10,0,1],"2015-09":[5,4,0,0],"2015-10":[7,6,0,0],"2015-11":[0,9,0,0],"2015-12":[0,8,0,0],
"2016-01":[3,8,1,0],"2016-02":[7,5,1,0],"2016-03":[0,8,0,0],"2016-04":[4,6,0,0],"2016-05":[2,8,0,0],"2016-06":[3,6,0,0],
"2016-07":[0,10,0,1], "2016-08":[1,8,0,1],"2016-09":[3,6,0,0],"2016-10":[7,6,0,0],"2016-11":[0,8,0,0],"2016-12":[1,8,0,0],"2017-01":[7,5,1,0]}

#读取数据，简单预处理
product_info = pd.read_csv("product_info.txt",index_col='product_id')
product_quantity = pd.read_csv("product_quantity.txt",)
product_quantity.sort_values(['product_id','product_date'],inplace=True)
train_day=product_quantity.groupby(['product_id','product_date']).sum()['ciiquantity'].unstack()

train_day.fillna(method='backfill',axis=1) 
train_day.fillna(method='ffill',axis=1) 

product_quantity['product_month']=product_quantity['product_date'].apply(lambda x: x[:7])
train_month = product_quantity.groupby(['product_id','product_month']).sum()['ciiquantity'].unstack()
train_index = train_month.index
#缺失值填补（均值填补）
#train_month.fillna(140,inplace=True) 
train_month = fillnan(train_month,1,30,140)#对大于12个月有空缺值的进行填补，其他的填补为140，取topK=30进行填补
#train_month_9month.fillna(140,inplace=True)
#对product_info进行数据预处理
product_info = preprocess(product_info) 
#%%
#根据均值和中位数找到销量大的波形
train_month_m = train_month.copy()
train_month_mean = train_month_m.mean(axis=1).sort_values(axis=0)
mean_index = train_month_mean.tail(2000).index
train_month_mid = train_month_m.median(axis=1).sort_values(axis=0)
median_index = train_month_mid.tail(2000).index
largeindex = [i for i in mean_index if i in median_index]                               

#%%
#绘制挑出曲线的波形
for index in index3:
    train_month[train_month.index == index].sum().plot(figsize=(12,6))
#%%
#针对单个ID训练模型
result_for_one = pd.DataFrame()
result_for_one = pd.concat(map(lambda x:train_for_one(train_month,product_info,x),largeindex),axis=1)
#result_for_one = result_for_one.iloc[23:,:]
#for index in largeindex:
    #result_for_one[[index]].plot(figsize=(12,6))
result_for_one = result_for_one.iloc[23:,:]
#%%
#对多个ID同时进行训练
result_for_some = pd.DataFrame()
result_for_some, feature_importance = train_for_some(train_month,product_info,train_month.index,'randomforest')

#for index in largeindex:
#    result_for_some[result_for_some.index == index].sum().plot(figsize=(12,6))
result_for_some = result_for_some.iloc[:,23:]
result_for_some = to_positive(result_for_some)
#%%
#采用均值的方法
average = pd.DataFrame(train_month.iloc[:,14:23].mean(axis=1),columns=['average_all']).reset_index()
submission=pd.read_csv('prediction_lilei_20170320.txt')
submission.shape
col=['product_id','product_month','ciiquantity_month']
submission.columns=col
#out=pd.merge(submission,average,on='product_id',how='left').fillna(132)
out=pd.merge(submission,average,on='product_id',how='left')
out.apply(lambda x: sum(x.isnull()))
out.ciiquantity_month=out.average_all
out.drop(['average_all'],axis=1,inplace=True)
#out.to_csv('sub_average9months132fillna2.txt',index=False)
#%%
#在均值的基础上加入部分模型预测（纵向）
for index in result_for_one.columns:
    v2 = list(result_for_one[index])
    v1 = list(out.loc[out.product_id==index,out.columns[2]])
    v0 = list(hyc_out.loc[hyc_out.product_id==index,hyc_out.columns[2]])
    v = list(map(lambda x: (x[0]+x[1]+x[2])/3, zip(v2, v1, v0)))
    out.loc[out.product_id==index,out.columns[2]] = v
#%%
#在均值的基础上加入部分模型预测（横向）
#输出结果融合
out1 = pd.read_csv('prediction_jice_20170417_167.97.txt')
for index in result_for_some.index:
    v2 = list(result_for_some.loc[index,:])
    v1 = list(out.loc[out.product_id==index,out.columns[2]])
    v = list(map(lambda x: (x[0]+x[1])/2, zip(v2, v1)))
    out.loc[out.product_id==index,out.columns[2]] = v2

#%%
#输出模型融合
out1 = pd.read_csv('prediction_jice_20170417_167.97.txt')
out2 = pd.read_csv('hyc_170.txt')
out1['ciiquantity_month'] = (out2['ciiquantity_month']+out1['ciiquantity_month'])/2








