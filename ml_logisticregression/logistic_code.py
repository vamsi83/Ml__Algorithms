import pandas as pd
import matplotlib.pyplot as mlp
import numpy as np

ds = pd.read_csv(r"C:\Users\vamsikrishna\OneDrive\Documents\projects\ml_logisticregression\logit classification.csv")
ds

X=ds.iloc[:,[2,3]].values
y=ds.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

print(classifier.get_params())

y_pred = classifier.predict(X_test)

bias=classifier.score(X_train,y_train)
print(bias)

var = classifier.score(X_test,y_test)
print(var)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac

bias=classifier.score(X_train,y_train)
print(bias)

var = classifier.score(X_test,y_test)
print(var)


# FUTURE PREDICTION ON VALIDATION DATA 

ds1=pd.read_csv(r"C:\Users\vamsikrishna\OneDrive\Documents\projects\ml_logisticregression\Future prediction1.csv")
ds1

ds2=ds1.copy()
ds2

ds1=ds1.iloc[:,[2,3]].values
ds1

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
m=sc.fit_transform(ds1)
m

y_pred1=pd.DataFrame()
y_pred1
 
ds2['y_pred1']=classifier.predict(m)
ds2.to_csv('final1.csv')
ds2