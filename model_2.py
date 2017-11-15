import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier


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
eq=clf.fit(x_train,y_train)
pred=eq.predict(x_test)
pred=np.ravel(pred)
# print(eq.score(x_train,y_train))

# 22,24,30,33,35
# cat4,cat6,cat12,cat15,cat17

sub = pd.read_csv('sample_submission.csv')
sub['target'] = pred
sub['target'] = sub['target'].astype(int)
sub.to_csv('sub_day2_2.csv', index=False)



# clf=RandomForestClassifier(max_depth=10,criterion='entropy')
# columns=[x for x in train.columns if x not in ['connection_id','target']]
# target=np.ravel(train['target'])

# for column in columns:
# 	clf.fit(train[column].values.reshape(-1,1),target)
# 	score=clf.score(train[column].values.reshape(-1,1),target)
# 	if(score>=0.70):
# 		print("{}".format(column))

# cont_1 : 0.605161038823
# cont_2 : 0.768822316856
# cont_3 : 0.752325656943
# cont_4 : 0.584358591198
# cont_5 : 0.586124613867
# cont_6 : 0.584364497629
# cont_7 : 0.585516251543
# cont_8 : 0.587867010815
# cont_9 : 0.587181864896
# cont_10 : 0.648880436131
# cont_11 : 0.634409681821
# cont_12 : 0.640971725918
# cont_13 : 0.702386788497
# cont_14 : 0.684697029656
# cont_15 : 0.591794786985
# cont_16 : 0.59403923051
# cont_17 : 0.589904729279
# cont_18 : 0.595521744523
# cat_1 : 0.62189395595
# cat_2 : 0.757765479277
# cat_3 : 0.5840278311
# cat_4 : 0.583956953936
# cat_5 : 0.583956953936
# cat_6 : 0.583962860366
# cat_7 : 0.584890169928
# cat_8 : 0.583956953936
# cat_9 : 0.724222861429
# cat_10 : 0.584051456821
# cat_11 : 0.584010111809
# cat_12 : 0.583980579657
# cat_13 : 0.584961047092
# cat_14 : 0.584423561932
# cat_15 : 0.584016018239
# cat_16 : 0.584913795649
# cat_17 : 0.583956953936
# cat_18 : 0.583962860366
# cat_19 : 0.584529877678
# cat_20 : 0.768905006881
# cat_21 : 0.664674230835
# cat_22 : 0.697236381248
# cat_23 : 0.645933127396

# cont_2 : 0.768786878274
# cont_3 : 0.752337469803
# cont_13 : 0.702386788497
# cat_2 : 0.757765479277
# cat_9 : 0.724222861429
# cat_20 : 0.768910913311

# cont_2 : 0.768798691135
# cont_3 : 0.752319750512
# cont_13 : 0.702386788497
# cont_14 : 0.684697029656
# cat_2 : 0.757765479277
# cat_9 : 0.724222861429
# cat_20 : 0.768910913311
# cat_21 : 0.664680137265
# cat_22 : 0.69723638124