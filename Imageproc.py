# -*- coding: utf-8 -*-
"""

@author: Zeinab
"""
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

data = load_digits()
X=data.data
y=data.target
idx=11
Im = X[idx,:]
Im=Im.reshape((8,8))
plt.imshow(Im)
print(y[idx])

X=X/16.0

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.svm import SVC
classifier = SVC(kernel='linear')
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print('cm',cm)
print("accuracy is=", np.diag(cm).sum()/np.sum(cm))

