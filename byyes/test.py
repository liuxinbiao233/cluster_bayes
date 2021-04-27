import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

iris=load_iris()
X,y=iris.data,iris.target
clf=GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(len(X_train))
print(len(X_test))
y_pred=clf.fit(X_train,y_train).predict(X_test)
print(clf.predict([[4.4, 3.2, 1.3, 0.2]]))
print(clf.score(X_test,y_test))
print("Number of mislabeled point out of a total %d points: %d" %(X_test.shape[0],(y_test!=y_pred).sum()))
