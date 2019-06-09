# -*- coding: utf-8 -*-
"""

@author: Zeinab
"""

import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import tensorflow as ts
import tensorflow.keras as keras

data = load_digits()
X=data.data
y=data.target

X=X/16.0

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

model = keras.Sequential()
model.add(keras.layers.Dense(64,input_shape=(64,),activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics = ['accuracy'])

model.fit(X_train, y_train,epochs=20)

model.evaluate(X_test,y_test)
