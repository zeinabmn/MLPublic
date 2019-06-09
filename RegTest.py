# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 20:59:56 2019

@author: Zeinab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes

data = load_diabetes()

X = data.data
y = data.target

print(np.mean(X))
print(np.var(X))

print(np.mean(y))
print(np.var(y))
#Missing data
if (np.isnan(X).sum()!=0):
	idx = np.argwhere(np.isnan(X)==True)
	X = np.delete(X[idx,:])
	y = np.delete(y[idx])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
	
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

err=np.sum((y_pred-y_test)**2)/(np.sum(y_test**2))
print('error is', err)

plt.scatter(X_train,y_train)
plt.plot(X_train, regressor.predict(X_train))