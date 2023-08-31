import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# veri yukleme
veriler = pd.read_csv('maas.csv')
veriler = veriler.drop("unvan",axis=1)
veriler = veriler.drop("Calisan ID",axis=1)




#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

X = veriler.iloc[:,0:3]
Y = veriler.iloc[:,-1:]

lin_reg.fit(X, Y)


tahmin = lin_reg.predict(X)
test = pd.DataFrame(np.array([[10,9,83]]))
tahmin2 = lin_reg.predict(test)

import statsmodels.api as sm

model = sm.OLS(lin_reg.predict(X), X)
print(model.fit().summary())


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
x1 = veriler.iloc[:, 0:1]
x2 = veriler.iloc[:, 2:3]
X = pd.concat([x1, x2], axis=1)
Y = veriler.iloc[:, -1:]

lin_reg.fit(X, Y)


tahmin = lin_reg.predict(X)
test = pd.DataFrame(np.array([[5, 50]]))
tahmin2 = lin_reg.predict(test)


import statsmodels.api as sm

model = sm.OLS(lin_reg.predict(X), X)
print(model.fit().summary())
print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))




