

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import numpy as np



boston=load_boston()
print("Keys of iris_dataset: \n{}".format(boston.keys()))
print(boston['feature_names'])
x=boston.data[:,0:2]
x1=np.square(boston.data[:,0:1])
y=boston.target
model=LinearRegression().fit(x,y)
print(model.intercept_)
print(model.coef_)
print(model.predict(x)[0]*1000)


