import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

remove_cols=["connection_id","target"]
cont_cols=["cont_1","cont_2","cont_3","cont_4","cont_5","cont_6","cont_7","cont_8","cont_9",\
			"cont_10","cont_11","cont_12","cont_13","cont_14","cont_15","cont_16","cont_17",\
			"cont_18"]
cat_cols=["cat_1","cat_2","cat_3","cat_4","cat_5","cat_6","cat_7","cat_8","cat_9","cat_10","cat_11",\
			"cat_12","cat_13","cat_14","cat_15","cat_16","cat_17","cat_18","cat_19","cat_20","cat_21",\
			"cat_22","cat_23"]
new_cols=['cont_2','cont_3','cont_13','cont_14','cat_2','cat_9','cat_20','cat_21','cat_22']

# print("taking train and test data")
# y_train=train["target"]
# x_cont_train=train.loc[:,cont_cols]
# x_cat_train=train.loc[:,cat_cols]

# print("training continuous model")
# clf_cont=RandomForestClassifier(max_depth=10)
# eq_cont=clf_cont.fit(x_cont_train,y_train)

# print("training categorical model")
# clf_cat=RandomForestClassifier(max_depth=10)
# eq_cat=clf_cat.fit(x_cat_train,y_train)

# x_train=train.drop(remove_cols,axis=1)
# x_test=test.drop(["connection_id"],axis=1)
# y_train=train["target"]

# print("print class score")
# print(eq_cat.score(x_cat_train,y_train))
# print(eq_cont.score(x_cont_train,y_train))
# print(eq.feature_importances_)


x_train=train.loc[:,new_cols]
x_train['cat_20_21']=x_train['cat_20']+x_train['cat_21']
x_train=x_train.drop(['cat_20','cat_21'],axis=1)
y_train=np.ravel(train["target"])

x_test=test.loc[:,new_cols]
x_test['cat_20_21']=x_test['cat_20']+x_test['cat_21']
x_test=x_test.drop(['cat_20','cat_21'],axis=1)
# x_test['cat_1_20_21']=x_test['cat_1']+x_test['cat_20']+x_test['cat_21']
# x_test=x_test.drop(['cat_1','cat_20','cat_21'],axis=1)



# print(x_test.head())
clf=RandomForestClassifier(criterion= 'entropy', max_depth=10)
# clf=XGBClassifier(max_depth=10)
x,xx,y,yy=train_test_split(x_train,y_train,test_size=0.2,random_state=10)
eq=clf.fit(x,y)
# pred=eq.predict(x_test)
# pred=np.ravel(pred)
print(eq.score(xx,yy))

# 22,24,30,33,35
# cat4,cat6,cat12,cat15,cat17

# sub = pd.read_csv('sample_submission.csv')
# sub['target'] = pred
# sub['target'] = sub['target'].astype(int)
# sub.to_csv('sub_day3_2.csv', index=False)



# clf=RandomForestClassifier(max_depth=10,criterion='entropy')
# columns=[x for x in train.columns if x not in ['connection_id','target']]
# target=np.ravel(train['target'])

# for column in columns:
# 	clf.fit(train[column].values.reshape(-1,1),target)
# 	score=clf.score(train[column].values.reshape(-1,1),target)
# 	if(score>=0.70):
# 		print("{}".format(column))
