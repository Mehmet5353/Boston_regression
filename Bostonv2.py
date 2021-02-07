from sklearn.linear_model import LinearRegression
from sklearn.datasets import  load_boston
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

boston=load_boston()
x=boston.data[:,0:1]
x1=boston.data[:,1:2]
x2=np.concatenate((x, x1), axis=1)
print(x)
print(x1)
print(x2)


y=boston.target

interaction=PolynomialFeatures(degree=2, include_bias=False )
                               


X=interaction.fit_transform(x2)
print(X)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)
model=LinearRegression().fit(X_train,y_train)

y_pred = model.predict(X_train)


print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
