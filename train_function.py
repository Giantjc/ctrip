# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:40:09 2017

@author: giantjc
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
#%%
def build_feature_for_one(*data):
    product_info1,train_month1 = data
    cols = product_info1.index.T
    colstime = ['startdate','upgradedate', 'cooperatedate']
    cols = [col for col in cols if col not in colstime]
    train_x = pd.DataFrame(columns=['month','holiday','weekend','issummer','iswinter','mean','median','variance','trend'])    
    for m in range(1,23): 
        train_tmp =  pd.DataFrame(index=[1])
        train_tmp['month'] = int(train_month1.index[m][5:7])#后期用one-hot处理                            
        train_tmp['holiday'] = knowledge[train_month1.index[m]][0]
        train_tmp['weekend'] = knowledge[train_month1.index[m]][1]
        train_tmp['issummer'] = knowledge[train_month1.index[m]][2]
        train_tmp['iswinter'] = knowledge[train_month1.index[m]][3]
        
        train_tmp['mean'] = train_month1[14:23].mean()
        train_tmp['variance'] = train_month1[14:23].std()
        train_tmp['median'] = train_month1[14:23].median()
        '''
        train_tmp['mean'] = train_month1[:m].mean()
        train_tmp['variance'] = train_month1[:m].std()
        train_tmp['median'] = train_month1[:m].median()
        '''
        if m in [1,2]:
            train_tmp['trend'] = train_month1[:m].mean()-train_month1[14:23].mean()
        else:
            train_tmp['trend'] = train_month1[m-2:m].mean()-train_month1[14:23].mean()
        '''
        for col in cols:
            train_tmp[col] = product_info1[col]
        '''
        train_x = train_x.append(train_tmp,ignore_index=True)  
    '''
    enc = OneHotEncoder()
    enc.fit([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]])
    tmp = enc.transform([[i]for i in train_x['month']]).toarray()
    train_x = pd.concat([train_x,pd.DataFrame(tmp)],axis=1)
    '''
    return train_x
def build_feature_for_predict(*data):
    product_info1,train_month1,month1 = data
    cols = product_info1.index.T
    colstime = ['startdate','upgradedate', 'cooperatedate']
    cols = [col for col in cols if col not in colstime]
    train_x = pd.DataFrame(columns=['month','holiday','weekend','issummer','iswinter','mean','median','variance','trend'])     
    train_tmp =  pd.DataFrame(index=[1])
    train_tmp['month'] = int(month1[5:7])
    train_tmp['holiday'] = knowledge[month1[:7]][0]
    train_tmp['weekend'] = knowledge[month1[:7]][1]
    train_tmp['issummer'] = knowledge[month1[:7]][2]
    train_tmp['iswinter'] = knowledge[month1[:7]][3]
    
    train_tmp['mean'] = train_month1[14:23].mean()
    train_tmp['variance'] = train_month1[14:23].std()
    train_tmp['median'] = train_month1[14:23].median()
    '''
    train_tmp['mean'] = train_month1[:-1].mean()
    train_tmp['variance'] = train_month1[:-1].std()
    train_tmp['median'] = train_month1[:-1].median()
    '''
    train_tmp['trend'] = train_month1[-3:-1].mean() - train_month1[14:23].mean()   
    '''
    for col in cols:
        train_tmp[col] = product_info1[col]
    '''
    train_x = train_x.append(train_tmp,ignore_index=True)
    '''
    #月份转为one hot编码
    enc = OneHotEncoder()
    enc.fit([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]])
    tmp = enc.transform([train_x['month']]).toarray()
    '''
    train_x = pd.concat([train_x,pd.DataFrame(tmp)],axis=1)  
    return train_x

def train_for_one(*data):
    train_month,product_info,index = data
    train_month_this = train_month.copy()
    data_y = train_month_this.loc[index,:][1:]        
    data_x = build_feature_for_one(product_info.loc[index,:],train_month_this.loc[index,:])     
    x_train = data_x
    y_train = data_y
    #x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.1,random_state=0)  
    xgtrain = xgb.DMatrix(x_train,label=y_train)
    #xgval = xgb.DMatrix(x_test,label=y_test)    
    params={}
    params['objective']='reg:linear'
    params['eta']=0.2
    params['min_child_weight']=2
    params['subsample']=0.7
    params['colsample_bytree']=0.7
    params['silent']=1
    params['max_depth']=5
    params['lambda']=0
    params['eval_metric']='rmse'
    #watchlist=[(xgval,'val'),(xgtrain,'train')]    
    xgboost_model=xgb.train(params,xgtrain,num_boost_round=180)  #,evals=watchlist)    
    months=['2015-12-01','2016-01-01','2016-02-01','2016-03-01','2016-04-01','2016-05-01','2016-06-01',
        '2016-07-01','2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01','2017-01-01']
    for month in months:
        feature_data=build_feature_for_predict(product_info.loc[index,:],train_month_this.loc[index,:],month)
        xgtest=xgb.DMatrix(feature_data)
        preds=xgboost_model.predict(xgtest,ntree_limit=xgboost_model.best_iteration)
        train_month_this.loc[index,month]=preds
    return train_month_this.loc[index,:]

#%%
def build_feature_for_some(*data):
    product_info1,train_month1 = data
    index1 = product_info1.index
    cols = product_info1.columns.T
    #colstime = ['startdate','upgradedate', 'cooperatedate']
    #cols = [col for col in cols if col not in colstime]
    train_x = pd.DataFrame(columns=['product_id','month','holiday','weekend','issummer','iswinter',*cols,'quantity'])    
    for m in range(1,23): 
        train_tmp =  pd.DataFrame(index = index1)        
        train_tmp['product_id'] = index1
        train_tmp['month'] = int(train_month1.columns[m][5:7])#后期用one-hot处理                            
        train_tmp['holiday'] = knowledge[train_month1.columns[m]][0]
        train_tmp['weekend'] = knowledge[train_month1.columns[m]][1]
        train_tmp['issummer'] = knowledge[train_month1.columns[m]][2]
        train_tmp['iswinter'] = knowledge[train_month1.columns[m]][3]
        #train_tmp['mean'] = train_month1.iloc[:,14:23].mean(axis=1)
        #train_tmp['variance'] = train_month1.iloc[:,14:23].std(axis=1)
        #train_tmp['median'] = train_month1.iloc[:,14:23].median(axis=1)                       
        train_tmp[cols] = product_info1.loc[:,cols]
        train_tmp['quantity'] = train_month1.iloc[:,m]
        '''
        if m in [1,2]:
            train_tmp['trend'] = train_month1.iloc[:,:m-1].mean()-train_month1.iloc[:,14:23].median(axis=1)
        else:
            train_tmp['trend'] = train_month1.iloc[:,m-3:m-1].mean()-train_month1.iloc[:,14:23].median(axis=1)         
        '''   
        train_x = train_x.append(train_tmp,ignore_index=True)  
    enc = OneHotEncoder()
    enc.fit([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]])
    tmp = enc.transform([[i]for i in train_x['month']]).toarray()
    train_x = pd.concat([train_x,pd.DataFrame(tmp)],axis=1)
    
    return train_x

def build_feature_for_some_predict(*data):
    product_info1,train_month1,month1 = data
    index1 = product_info1.index
    cols = product_info1.columns.T
    #colstime = ['startdate','upgradedate', 'cooperatedate']
    #cols = [col for col in cols if col not in colstime]
    train_x = pd.DataFrame(columns=['product_id','month','holiday','weekend','issummer','iswinter',*cols])
    train_tmp =  pd.DataFrame(index = index1)       
    train_tmp['product_id'] = index1
    train_tmp['month'] = int(month1[5:7])#后期用one-hot处理                            
    train_tmp['holiday'] = knowledge[month1[:7]][0]
    train_tmp['weekend'] = knowledge[month1[:7]][1]
    train_tmp['issummer'] = knowledge[month1[:7]][2]
    train_tmp['iswinter'] = knowledge[month1[:7]][3]
    #train_tmp['mean'] = train_month1.iloc[:,14:23].mean(axis=1)
    #train_tmp['variance'] = train_month1.iloc[:,14:23].std(axis=1)
    #train_tmp['median'] = train_month1.iloc[:,14:23].median(axis=1)                 
    train_tmp[cols] = product_info1.loc[:,cols]
    #train_tmp['trend'] = train_month1.iloc[:,-3:-1].mean() - train_month1.iloc[:,14:23].median(axis=1)               
    train_x = train_x.append(train_tmp,ignore_index=True) 
    
    enc = OneHotEncoder()
    enc.fit([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]])
    tmp = enc.transform([[i]for i in train_x['month']]).toarray()

    train_x = pd.concat([train_x,pd.DataFrame(tmp)],axis=1) 
    
    return train_x

#给定数据，输出训练数据
def build_traindata(*data):
    pre_index,product_info,train_month_this = data
    train_data = build_feature_for_some(product_info.loc[pre_index,:],train_month_this.loc[pre_index,:])
    cols = [col for col in train_data.columns if col not in ['quantity']]    
    y_data = train_data.loc[:,['quantity']]
    x_data = train_data.loc[:,cols]
    return y_data,x_data

def train_for_some(*data):
    train_month,product_info,index,model = data
    train_month_this = train_month.copy()
    '''
    #测试机和训练集的构造是先对商品进行划分，再对划分后的商品分别生成样本
    pre_train_index,pre_test_index = train_test_split(index,test_size=0.2,random_state=0)
    y_train,x_train = build_traindata(pre_train_index,product_info,train_month_this)
    y_test,x_test = build_traindata(pre_test_index,product_info,train_month_this)      
    '''
    #开始使用的方法，直接全部生成训练样本，然后再随机划分训练集和测试集
    data_x = build_feature_for_some(product_info.loc[index,:],train_month_this.loc[index,:])
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
        params={}
        params['objective']='reg:linear'
        params['eta']=0.2  #0.2尝试的较优
        params['min_child_weight']=8
        params['subsample']=0.8
        params['colsample_bytree']=0.7
        params['silent']=1
        params['max_depth']=8
        params['lambda']=0
        params['eval_metric']='rmse'
        watchlist=[(xgval,'val'),(xgtrain,'train')]    
        xgboost_model=xgb.train(params,xgtrain,num_boost_round=4000,evals=watchlist)           
        for month in months:
            feature_data=build_feature_for_some_predict(product_info.loc[index,:],train_month_this.loc[index,:],month)
            xgtest=xgb.DMatrix(feature_data)
            preds=xgboost_model.predict(xgtest,ntree_limit=xgboost_model.best_iteration)
            preds=list(preds)
            train_month_this.loc[index,month]=preds
                                
    #随机森林进行训练                            
    elif model == 'randomforest':
        params = dict(n_estimators = 800,criterion='mse',max_depth=None,min_samples_split=6)#,min_samples_leaf=8,)      
        rf = RandomForestRegressor(**params)
        rf.fit(x_train,y_train)
        
        feature_importance = pd.DataFrame([x_train.columns,rf.feature_importances_])

        for month in months:
            feature_data=build_feature_for_some_predict(product_info.loc[index,:],train_month_this.loc[index,:],month)
            preds = rf.predict(feature_data)
            preds=list(preds)
            train_month_this.loc[index,month]=preds
                               
    #神经网络进行训练 
    elif model == 'NN':
        
        print("waiting")
        
    return train_month_this.loc[index,:],feature_importance.T

#%%
def change(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1
def build_feature(*data):
    product_info,train_month_tmp = data
    train_x=pd.DataFrame()
    cols0 = product_info.columns 
    colstime = ['startdate','upgradedate', 'cooperatedate']
    cols0 = [col for col in cols0 if col not in colstime]
    #product_ID信息
    for col in cols0:
        train_x[col]=product_info[col]
    #统计信息
    train_x['three_avg'] = train_month_tmp.iloc[:,-3:-1].mean(axis=1)
    train_x['mean'] = train_month_tmp.mean(axis=1)
    train_x['mid'] = train_month_tmp.median(axis = 1)
    #train_x['month'] = train_month_tmp.columns[-1][5:]
           
    train_x.index=product_info.index
    cols1=['q1','q2','q3','q4','q5','q6','q7','q8','q9','q10','q11','q12']
    cols2=['df1','df2','df3','df4','df5','df6']
    for i,m in enumerate(cols1):
        train_x[m]=train_month_tmp.iloc[:,-1-i]
    for j,n in enumerate(cols2):
    	train_x[n]=train_month_tmp.iloc[:,-1-j] - train_month_tmp.iloc[:,-1-j-1]
    #for j,n in enumerate(cols2):
    	#train_x[n]=train_x[n].apply(lambda x:change(x))
    return train_x