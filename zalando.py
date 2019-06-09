# -*- coding: utf-8 -*-
"""

@author: Zeinab
"""

import numpy as np

import matplotlib.pyplot as plt
import tensorflow as ts
import tensorflow.keras as keras

data = keras.datasets.fashion_mnist
(X_train,y_train),(X_test,y_test)=data.load_data()


X_train=X_train/255.0
X_test=X_test/255.0

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128,activation='relu'))
#model.add(keras.layers.Dense(32,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics = ['accuracy'])

model.fit(X_train, y_train,epochs=10,batch_size=128)

print(model.evaluate(X_test,y_test))

