# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:40:09 2017

@author: giantjc
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb

from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from datetime import datetime
from dateutil.parser import parse

import data_preprocessing as prep

#人工经验统计（节假日天数，周末天数(除节假日)，是否寒暑假）
knowledge = {
"2014-01":[2,7,1,0],"2014-02":[6,5,1,0],"2014-03":[0,10,0,0],"2014-04":[3,6,0,0],"2014-05":[4,6,0,0],"2014-06":[2,8,0,0],
"2014-07":[0,8,0,1],"2014-08":[1,10,0,1],"2014-09":[3,5,0,0],"2014-10":[7,5,0,0],"2014-11":[0,10,0,0],"2014-12":[0,8,0,0],
"2015-01":[3,7,1,0],"2015-02":[7,4,1,0],"2015-03":[0,9,0,0],"2015-04":[3,6,0,0],"2015-05":[3,8,0,0],"2015-06":[3,6,0,0],
"2015-07":[0,8,0,1],"2015-08":[1,10,0,1],"2015-09":[5,4,0,0],"2015-10":[7,6,0,0],"2015-11":[0,9,0,0],"2015-12":[0,8,0,0],
"2016-01":[3,8,1,0],"2016-02":[7,5,1,0],"2016-03":[0,8,0,0],"2016-04":[4,6,0,0],"2016-05":[2,8,0,0],"2016-06":[3,6,0,0],
"2016-07":[0,10,0,1], "2016-08":[1,8,0,1],"2016-09":[3,6,0,0],"2016-10":[7,6,0,0],"2016-11":[0,8,0,0],"2016-12":[1,8,0,0],"2017-01":[7,5,1,0]}

def build_feature_for_some(*data):
    product_info1,train_month1,product_price = data
    index1 = product_info1.index
    cols = product_info1.columns.T
    #构造特征矩阵
    train_x = pd.DataFrame()
    for m in range(0,23): 
        train_tmp =  pd.DataFrame(index = index1)        
        train_tmp['product_id'] = index1
        train_tmp['month'] = int(train_month1.columns[m][5:7])#后期用one-hot处理   
        month_num = prep.trans2time(train_month1.columns[m])                         
        train_tmp['holiday'] = knowledge[train_month1.columns[m]][0]
        train_tmp['weekend'] = knowledge[train_month1.columns[m]][1]
        train_tmp['issummer'] = knowledge[train_month1.columns[m]][2]
        train_tmp['iswinter'] = knowledge[train_month1.columns[m]][3]
        train_tmp['delttime1'] = product_info1.loc[:,'startdate']-month_num
        train_tmp['delttime2'] = product_info1.loc[:,'upgradedate']-month_num
        train_tmp['delttime3'] = product_info1.loc[:,'cooperatedate']-month_num  
        train_tmp['price'] = product_price[index1]                   
        train_tmp[cols] = product_info1.loc[:,cols]        
        #组合的特征
        train_tmp['combinefeature1'] = train_tmp['delttime1']*train_tmp['delttime2']
        train_tmp['combinefeature2'] = train_tmp['delttime1']*train_tmp['delttime3']
        train_tmp['combinefeature3'] = train_tmp['delttime2']*train_tmp['delttime3']
        train_tmp['eval_all'] = prep.to_positive(train_tmp['eval']*train_tmp['eval2']*train_tmp['eval3']*train_tmp['eval4'])               
        train_tmp['quantity'] = train_month1.iloc[:,m]  
        #将含有nan值的样本剔除掉         
        train_x = train_x.append(train_tmp,ignore_index=True)        
    enc = OneHotEncoder()
    enc.fit([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]])
    tmp = enc.transform([[i]for i in train_x['month']]).toarray()
    train_x = pd.concat([train_x,pd.DataFrame(tmp)],axis=1)        
    return train_x

def build_feature_for_some_predict(*data):
    product_info1,train_month1,month1,product_price = data
    index1 = product_info1.index
    cols = product_info1.columns.T
    #构造特征矩阵
    train_x = pd.DataFrame()
    train_tmp =  pd.DataFrame(index = index1)       
    train_tmp['product_id'] = index1
    train_tmp['month'] = int(month1[5:7])#后期用one-hot处理  
    month_num = prep.trans2time(month1)                            
    train_tmp['holiday'] = knowledge[month1[:7]][0]
    train_tmp['weekend'] = knowledge[month1[:7]][1]
    train_tmp['issummer'] = knowledge[month1[:7]][2]
    train_tmp['iswinter'] = knowledge[month1[:7]][3]
    train_tmp['delttime1'] = product_info1.loc[:,'startdate']-month_num
    train_tmp['delttime2'] = product_info1.loc[:,'upgradedate']-month_num
    train_tmp['delttime3'] = product_info1.loc[:,'cooperatedate']-month_num
    train_tmp['price'] = product_price[index1]         
    train_tmp[cols] = product_info1.loc[:,cols]    
    #组合的特征
    train_tmp['combinefeature1'] = train_tmp['delttime1']*train_tmp['delttime2']#三个组合中combinefeature1重要度最大
    train_tmp['combinefeature2'] = train_tmp['delttime1']*train_tmp['delttime3']
    train_tmp['combinefeature3'] = train_tmp['delttime2']*train_tmp['delttime3']               
    train_tmp['eval_all'] = prep.to_positive(train_tmp['eval']*train_tmp['eval2']*train_tmp['eval3']*train_tmp['eval4'])                 
    train_x = train_x.append(train_tmp,ignore_index=True)     
    enc = OneHotEncoder()
    enc.fit([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]])
    tmp = enc.transform([[i]for i in train_x['month']]).toarray()
    train_x = pd.concat([train_x,pd.DataFrame(tmp)],axis=1)        
    return train_x

#对所有的商品样本同时训练模型
def train_for_some(*data):
    train_month,product_info,index,product_price,model = data
    train_month_this = train_month.copy() 
    #直接全部生成训练样本，然后再随机划分训练集和测试集
    data_x = build_feature_for_some(product_info.loc[index,:],train_month_this.loc[index,:],product_price)
    cols = [col for col in data_x.columns if col not in ['quantity']]    
    data_y = data_x.loc[:,['quantity']]
    data_x = data_x.loc[:,cols]
    x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.2,random_state=0) 
    months=['2015-12-01','2016-01-01','2016-02-01','2016-03-01','2016-04-01','2016-05-01','2016-06-01',
            '2016-07-01','2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01','2017-01-01']
    #xgboost进行训练
    if model == 'xgboost':
        xgtrain = xgb.DMatrix(x_train,label=y_train)
        xgval = xgb.DMatrix(x_test,label=y_test)
        feature_importance = pd.DataFrame()
        params={}
        params['objective']='reg:linear'
        params['eta']=0.20  #0.2尝试的较优
        params['min_child_weight']=6
        params['subsample']=0.8
        params['colsample_bytree']=0.7
        params['silent']=1
        params['max_depth']=8 #164.8树的深度用8跑出来的
        params['lambda']=0
        params['eval_metric']='rmse'
        watchlist=[(xgval,'val'),(xgtrain,'train')]    
        xgboost_model=xgb.train(params,xgtrain,num_boost_round=6000,evals=watchlist)
        xg_test = xgb.DMatrix(x_test)
        y_pred = xgboost_model.predict(xg_test,ntree_limit=xgboost_model.best_iteration)
        print (mean_squared_error(y_pred,y_test))          
        for month in months:
            feature_data=build_feature_for_some_predict(product_info.loc[index,:],train_month_this.loc[index,:],month,product_price)
            xgtest=xgb.DMatrix(feature_data)
            preds=xgboost_model.predict(xgtest,ntree_limit=xgboost_model.best_iteration)
            preds=list(preds)
            train_month_this.loc[index,month]=preds                               
    #随机森林进行训练                            
    elif model == 'rf':
        params = dict(n_estimators = 200,criterion='mse',max_depth=None,min_samples_split=6)#,min_samples_leaf=8,)      
        rf = RandomForestRegressor(**params)
        rf.fit(x_train,y_train)
        #rf.fit(data_x,data_y)
        y_pred = rf.predict(x_test)
        print (mean_squared_error(y_pred,y_test))
        feature_importance = pd.DataFrame([x_train.columns,rf.feature_importances_])
        for month in months:
            feature_data=build_feature_for_some_predict(product_info.loc[index,:],train_month_this.loc[index,:],month,product_price)
            preds = rf.predict(feature_data)
            preds=list(preds)
            train_month_this.loc[index,month]=preds  
    #ExtraTrees模型训练
    elif model == 'et':
        params = dict(n_estimators = 200,criterion='mse',max_depth=None,min_samples_split=6)#,min_samples_leaf=8,)      
        et = ExtraTreesRegressor(**params)
        et.fit(x_train,y_train)
        y_pred = et.predict(x_test)
        print (mean_squared_error(y_pred,y_test))
        feature_importance = pd.DataFrame([x_train.columns,et.feature_importances_])
        for month in months:
            feature_data=build_feature_for_some_predict(product_info.loc[index,:],train_month_this.loc[index,:],month,product_price)
            preds = et.predict(feature_data)
            preds=list(preds)
            train_month_this.loc[index,month]=preds
    #GBDT模型训练                            
    elif model == 'gbdt':
        params = dict(loss = 'ls',n_estimators = 3000,learning_rate = 0.1)
        gbdt = GradientBoostingRegressor(**params)
        gbdt.fit(x_train,y_train)
        y_pred = gbdt.predict(x_test)
        print (mean_squared_error(y_pred,y_test))
        feature_importance = pd.DataFrame([x_train.columns,gbdt.feature_importances_])
        for month in months:
            feature_data=build_feature_for_some_predict(product_info.loc[index,:],train_month_this.loc[index,:],month,product_price)
            preds = gbdt.predict(feature_data)
            preds=list(preds)
            train_month_this.loc[index,month]=preds                                 
                                
    return train_month_this.loc[index,:],feature_importance.T

#时间序列模型训练
def Arma_model(train_month,p,q):
    n = 0;
    TempData = pd.DataFrame()
    last_result = pd.DataFrame(index = pd.date_range('2015-12-01','2017-02-01',freq='M'))
    final_result = pd.DataFrame(index = pd.date_range('2015-12-01','2017-02-01',freq='M'))
    for index in train_month.index:
        n = n + 1
        TempData[index] = pd.ewma(train_month.loc[index],10)
        ArmaModel = sm.tsa.ARMA(TempData[index],(p,q)).fit()
        predict_series = ArmaModel.predict('2015-12','2017-01',dynamic = True)        
        predict_one = pd.DataFrame(predict_series.values,columns = [index],index = pd.date_range('2015-12-01','2017-02-01',freq='M'))
        final_result = pd.concat([last_result,predict_one],axis=1)
        last_result = final_result
        print('the %d counts is calculated'%(n))      
    final_result.index = final_result.index.strftime("%Y-%m")
    final_result = final_result.T   
    return final_result