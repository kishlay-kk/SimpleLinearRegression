# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:34:25 2018

@author: kishl
"""
# Importing Libraries 
import numpy as nm
import pandas as pd
import matplotlib.pyplot as plot

# Importing Data
data= pd.read_csv('Salary_Data.csv')

# Separation into Dependent and Independent Values 
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values

# Separation into Training Set and Test Set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.33)

# Fitting  Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Predicting the Test Set Results
y_pred = regressor.predict(x_test)

# Visualising the  Training and Test Set Result
plot.scatter(x_train,y_train, color='red')
plot.plot(x_train, regressor.predict(x_train),color = 'blue')
plot.title("Salary Vs Experience(Training Set)")
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()

plot.scatter(x_test,y_test , color='red')
plot.plot(x_train, regressor.predict(x_train),color = 'blue')
plot.title("Salary Vs Experience(Test Set)")
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()
