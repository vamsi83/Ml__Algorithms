import pandas as pd
import matplotlib.pyplot as mlp
import numpy as np

ds = pd.read_csv(r"C:\Users\vamsikrishna\OneDrive\Documents\projects\ml_nb\logit classification.csv")
ds

X=ds.iloc[:,[2,3]].values
y=ds.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import Normalizer
sc=Normalizer()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

'''from sklearn.naive_bayes import BernoulliNB
classifier=BernoulliNB()
classifier.fit(X_train, y_train)'''

'''from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train, y_train)'''

from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
print(ac)

bias=classifier.score(X_train,y_train)
print(bias)

var=classifier.score(X_test,y_test)
print(var)
