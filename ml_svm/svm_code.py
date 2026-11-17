import pandas as pd
import matplotlib.pyplot as mlp
import numpy as np

ds = pd.read_csv(r"C:\Users\vamsikrishna\OneDrive\Documents\projects\ml_svm\logit classification.csv")
ds

X=ds.iloc[:,[2,3]].values
y=ds.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.svm import SVC
classifier_svc=SVC(kernel="sigmoid",gamma="scale",degree=5)
classifier_svc.fit(X_train,y_train)

from sklearn.neighbors import KNeighborsClassifier
classifier_knn=KNeighborsClassifier
classifier_knn.fit(X_train, y_train)

y_pred=classifier_svc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
print(ac)

bias=classifier_svc.score(X_train,y_train)
print(bias)

var=classifier_svc.score(X_test,y_test)
print(var)

