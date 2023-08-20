# -*- coding: utf-8 -*-
"""Task 1


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

modellr= LinearRegression()
modellr.fit(X_train, y_train)

y_pred = modellr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

print("Coefficient:", modellr.coef_[0])

print("Intercept:", modellr.intercept_)
