# -*- coding: utf-8 -*-
"""

@author: Zeinab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Loading data
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data["data"]
y = data["target"]
print(X.shape, y.shape)
type(X)

#Missing data
if (np.isnan(X).sum()!=0):
	idx = np.argwhere(np.isnan(X)==True)
	X = np.delete(X[idx,:])
	y = np.delete(y[idx])
	
	
#Splitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
print(np.mean(X_train))
print(np.var(X_train))

#Scaling	
from sklearn.preprocessing import StandardScaler
stdScalar = StandardScaler()
X_train = stdScalar.fit_transform(X_train)
X_test = stdScalar.transform(X_test)
print(np.mean(X_train))
print(np.var(X_train))

#Dimension reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
pca.explained_variance_ratio_

"""#visualising X_train
plt.figure()
idx_0 = (y_train==0)
idx_1 = (y_train==1)
plt.scatter(X_train[idx_0,0],X_train[idx_0,1])
plt.scatter(X_train[idx_1,0],X_train[idx_1,1])
plt.show()"""

#classification
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

#prediction
y_pred=classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("confusion matrix is = ",cm)
print("accuracy is=", np.diag(cm).sum()/np.sum(cm))

#PC dependency to the original features

plt.matshow(pca.components_,cmap='viridis')
plt.colorbar()
plt.yticks([0,1],['PC1','PC2'])
plt.xticks(range(len(data.feature_names)),data.feature_names,rotation = 65)
plt.show()

# Visualising the Training set results
from matplotlib.colors import ListedColormap
plt.figure()
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('Orange', 'Aqua'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
plt.figure()
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('Orange', 'Aqua'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


