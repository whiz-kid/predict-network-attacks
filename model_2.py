import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
new_cols=['cont_2','cont_3','cont_13','cont_14','cat_2','cat_9','cat_20','cat_21','cat_22']
x_train=train.loc[:,new_cols]
x_train['cat_20_21']=x_train['cat_20']+x_train['cat_21']
x_train=x_train.drop(['cat_20','cat_21'],axis=1)
y_train=np.ravel(train["target"])

x_test=test.loc[:,new_cols]
x_test['cat_20_21']=x_test['cat_20']+x_test['cat_21']
x_test=x_test.drop(['cat_20','cat_21'],axis=1)
params={
    "objective":"multi:softmax",     
    "learning_rate":0.1,
    "subsample":0.8,
    "colsample_bytree": 0.8,
#     'eval_metric':'auc',
    "max_depth":6,
    'silent':1,
    'nthread':3,
    'num_class':3
        
       } 

x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)

kf=KFold(n=len(x_train),n_folds=10,random_state=100)
scores=[]
predictions=[]
for trainkf,testkf in kf:
    train_data_x,train_data_y=x_train[trainkf],y_train[trainkf]
    valid_data_x,valid_data_y=x_train[testkf],y_train[testkf]
    
    dtrain=xgb.DMatrix(train_data_x,train_data_y)
    dvalid=xgb.DMatrix(valid_data_x,valid_data_y)
    dtest=xgb.DMatrix(x_test)
    watchlist=[(dtrain,'train'),(dvalid,'valid')]
    gbm=xgb.train(params,dtrain,6000,evals=watchlist,early_stopping_rounds=50,verbose_eval=1)
    cv=gbm.predict(dvalid)
    lst=gbm.predict(dtest)
    scores.extend(cv)
    predictions.append(lst)

from scipy.stats import mode
pred=mode(predictions,axis=0)
pred=pred[0][0]
pred=np.ravel(pred)
sub = pd.read_csv('sample_submission.csv')
sub['target'] = pred
sub['target'] = sub['target'].astype(int)
sub.to_csv('sub_day4_1.csv', index=False)