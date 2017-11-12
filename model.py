import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

remove_cols=["connection_id","target"]
cont_cols=["cont_1","cont_2","cont_3","cont_4","cont_5","cont_6","cont_7","cont_8","cont_9",\
			"cont_10","cont_11","cont_12","cont_13","cont_14","cont_15","cont_16","cont_17",\
			"cont_18"]
cat_cols=["cat_1","cat_2","cat_2","cat_4","cat_5","cat_6","cat_7","cat_8","cat_9","cat_10","cat_11",\
			"cat_12","cat_13","cat_14","cat_15","cat_16","cat_17","cat_18","cat_19","cat_20","cat_21",\
			"cat_22","cat_23"]

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

x_train=train.drop(remove_cols,axis=1)
x_test=test.drop(["connection_id"],axis=1)
y_train=train["target"]

# print("print class score")
# print(eq_cat.score(x_cat_train,y_train))
# print(eq_cont.score(x_cont_train,y_train))
# print(eq.feature_importances_)

clf=RandomForestClassifier(min_samples_split= 2, min_weight_fraction_leaf= 0, \
	criterion= 'entropy', max_depth=15, min_samples_leaf= 10)
eq=clf.fit(x_train,y_train)
pred=eq.predict(x_test)
pred=np.ravel(pred)
# print(eq.score(x_train,y_train))

# 22,24,30,33,35
# cat4,cat6,cat12,cat15,cat17

sub = pd.read_csv('sample_submission.csv')
sub['target'] = pred
sub['target'] = sub['target'].astype(int)
sub.to_csv('sub1.csv', index=False)
