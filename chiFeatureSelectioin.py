import scipy 
import sklearn
import xgboost 
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.stats.stats import  pearsonr
from sklearn.preprocessing import Imputer
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import chi2 
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pylab import rcParamsfrom sklearn.svm import SVC

# Loading features
a_features_df = pd.read_csv('rFeatures.csv',header=None)
# Adding header
a_features_df.columns = [i for i in range(1,54838)]
# loading ptype data
ptype_df = pd.read_csv('RMaps_A_Ptyp.csv',encoding='latin-1')
# selecting 7 ptype features
ptype_features = ['GE_S ','X','STATUS','AND_CATEGORY','F','P','V']


X =a_features_df

# combining Ptype features with rfeatures
for feature in ptype_features:
    X.insert(0,feature,ptype_df[feature])
    
y =ptype_df[['DX_GROUP']]
y = sklearn.utils.validation.column_or_1d(y, warn=True)

# missing values handlimg mising values
X.fillna(a_features_df.mean(), inplace=True)

# Feature selection
chi2_features = SelectKBest(chi2, k = 250) 
X = chi2_features.fit_transform(X, y)

# test train splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""Logistic regression"""

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#accuracy
a1=classifier.score(X_test,y_test)
print("Accuracy Logistic regression ",a1)


"""KNN"""

# Fitting K-NN to the Training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#accuracy
a2=classifier.score(X_test,y_test)
print("Accuracy KNN",a2)

"""SVM"""

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#accuracy
a3=classifier.score(X_test,y_test)
print("Accuracy SVM",a3)


"""Kernel SVM"""

# Fitting Kernel SVM to the Training set
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#accuracy
a4=classifier.score(X_test,y_test)
print("Accuracy Kernel SVM",a4)

"""Naive bayes"""

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#accuracy
a5=classifier.score(X_test,y_test)
print("Accuracy Naive bayes ",a5)

"""Random forest"""

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#accuracy
a6=classifier.score(X_test,y_test)
print("Accuracy Random forest",a6)


"""Decision tree"""

# Fitting Decision Tree Classification to the Training set
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#accuracy
a7=classifier.score(X_test,y_test)
print("Accuracy Decision tree ",a7)


"""XGBoost"""
classifier = xgboost.XGBClassifier()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
a8 = accuracy_score(y_test, y_pred)
print("Accuracy XGBoost ",a8)






