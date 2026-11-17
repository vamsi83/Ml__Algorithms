import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

 
dataset = pd.read_csv(r'C:\Users\vamsikrishna\Downloads\Salary_Data.csv')

x = dataset.iloc[:,:-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) 

y_pred = regressor.predict(x_test)


comparision = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})
print(comparision) 

plt.scatter(x_test, y_test, color = 'Red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary of employee based on experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# validataion or future data 

c_inter =regressor.intercept_
print(f'Intercept: {regressor.intercept_}')

m_coef = regressor.coef_
print(f'Coefficient: {regressor.coef_}')

y_12 = m_coef*12 + c_inter
print(y_12)

y_20 = m_coef*20 + c_inter
print(y_20) 

bias_training = regressor.score(x_train, y_train)
print(bias_training) 

variance_testing = regressor.score(x_test,y_test)
print(variance_testing)

import pickle 

# Save the trained model to disk
filename = 'linear_regression_model.pkl'

# Open a file in write-binary mode and dump the model
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)

print("Model has been pickled and saved as linear_regression_model.pkl")

import os
os.getcwd()







